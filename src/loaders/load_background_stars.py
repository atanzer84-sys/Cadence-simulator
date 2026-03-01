import logging
import numpy as np
import astropy.units as u
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from astropy.table import Table
from loaders.run_waltzer_context import get_repo_root
from astropy.coordinates import SkyCoord
from astropy.table import join, vstack as table_vstack
from astroquery.gaia import Gaia
from configs.global_config import GlobalConfig
from domain.star_catalog import StarCatalog
from loaders.load_stellar_and_planetary_properties import load_excel_mapping , infer_mamajek, apply_log_r_fallback, get_missing_properties
from loaders.load_gaia import get_gaia_stellar_properties
from flux.flux_calc import calculateFluxOnEarth


def lookup_background_stars(ctx: RunContext, cfg: GlobalConfig, star: Star):
    table = load_background_csv_if_exists(star)
    if table is None:
        table = gaia_lookup_for_background_stars(star)
        if table is not None and len(table) > 0:
            save_background_stars_csv(table, ctx.output_dir, star.name)
    if table is None or len(table) == 0:
        return StarCatalog()

    catalog = create_background_star_catalog(table)

    total = len(catalog.stars_by_id)
    logging.info("Starting flux calculation for %d background stars", total)
    print(f"==== STARTING FLUX CALCULATION FOR {total} BACKGROUND STARS =====")
    
    for i, (star_id, bg_star) in enumerate(catalog.stars_by_id.items(), start=1):
        logging.info("Calculating Flux on Earth %d/%d for %s", i, total, star_id)

        print(f"Flux {i}/{total} for {star_id}")
        flux_unred, wavelengths = calculateFluxOnEarth(bg_star, ctx)
        catalog.flux_earth_by_id[star_id] = (wavelengths, flux_unred)    

    return catalog
def load_background_csv_if_exists(star: Star) -> Table | None:
    repo_root = get_repo_root()
    csv_name = star.name.replace(" ", "_")
    csv_path = repo_root / "data" / "BackgroundStars" / f"{csv_name}.csv"

    if not csv_path.exists():
        return None

    table = Table.read(csv_path, format="csv")

    logging.info("Background stars: loading cached CSV: %s", csv_path)
    logging.info("Loaded background CSV for %s: rows=%d, columns=%s", star.name, len(table), list(table.colnames))
    return table

def create_background_star_catalog(table):
    
    catalog = StarCatalog()
    required_keys = load_required_stellar_parameters()

    for row in table:

        star_params = get_gaia_stellar_properties(row, log_output=False)
        # TODO: MAGNITUDE CUTOFF
        if not _passes_magnitude_cutoff(star_params):
            continue

        _set_background_star_name(star_params, row)
        _apply_distance_from_parallax_if_missing(star_params, row)

        if star_params.get("effective_temperature") is not None:
            star_params = infer_mamajek(star_params, log_output=False)
        
        if star_params.get("radius") is not None:
            star_params = apply_log_r_fallback(star_params, log_output=False)

        if not _ensure_required_properties(star_params, required_keys):
            continue

        bg_star = Star.from_params(star_params, required_keys, log_output = False)

        catalog.add_star(bg_star.name, bg_star)

    return catalog

def load_required_stellar_parameters():
    mapping = load_excel_mapping()
    return mapping["required_stellar_parameters"]


def save_background_stars_csv(table: Table, output_dir, star_name: str) -> None:
    """Write background stars table to CSV in output_dir. Use same name as cache so it can be moved to data/BackgroundStars/."""
    csv_name = star_name.replace(" ", "_")
    csv_path = output_dir / f"{csv_name}.csv"
    table.write(csv_path, format="ascii.csv", overwrite=True)
    msg = f"Background stars: saved to {csv_path} (move to data/BackgroundStars/ for cache)"
    logging.info(msg)
    print(msg)


def _passes_magnitude_cutoff(star_params: dict, max_mag: float = 20.0) -> bool:
    """True if star has gaia_magnitude and it is <= max_mag."""
    mag = star_params.get("gaia_magnitude")
    if mag is None:
        return False
    return float(mag) <= max_mag


def _set_background_star_name(star_params: dict, row) -> None:
    """Set star_params['name'] from row source_id, or '0000' if missing."""
    if "source_id" in row.colnames and row["source_id"] is not None:
        star_params["name"] = f"gaia_{int(row['source_id'])}"
    else:
        star_params["name"] = "0000"


def _apply_distance_from_parallax_if_missing(star_params: dict, row) -> None:
    """If star_params has no distance, set it from row parallax (distance_pc = 1000/parallax_mas) when valid."""
    if star_params.get("distance") is not None:
        return
    if "parallax" not in row.colnames:
        return
    par = row["parallax"]
    if par is None or np.ma.is_masked(par) or not np.isfinite(par) or float(par) <= 0.0:
        return
    star_params["distance"] = 1000.0 / float(par)


def _ensure_required_properties(star_params: dict, required_keys: list[str]) -> bool:
    """Return False if any required keys are missing (logs and skip). Otherwise normalize distance key and return True."""
    missing = get_missing_properties(star_params, required_keys, log_output=False)
    if missing:
        logging.info("Background star %s skipped. Missing: %s", star_params.get("name"), missing)
        return False
    if "distance" in star_params:
        star_params["distance"] = star_params.pop("distance")
    return True







def gaia_lookup_for_background_stars(star: Star) -> Table | None:
    """
    Fetch background stars from Gaia in a cone around the target star.

    Runs a cone search, removes the central (target) star, queries
    astrophysical_parameters for the rest, and returns the joined table.
    Used when no cached CSV exists in data/BackgroundStars/.

    Parameters
    ----------
    star : Star
        Target star; must have right_ascension and declination.

    Returns
    -------
    Table | None
        Table of field stars (cone + AP columns), or None if no sources,
        no field after removing central, or on Gaia/TAP error.
    """
    radius_arcsec = 150.0
    # g_mag_limit: float | None = 18.0  # set to None to disable magnitude filter
    g_mag_limit = None

    print("==== Gaia background search: START =====")
    logging.info("Gaia background search: START")

    center = SkyCoord(ra=star.right_ascension * u.deg, dec=star.declination * u.deg, frame="icrs")

    cone_small = _gaia_cone_search(center, radius_arcsec, g_mag_limit)
    if cone_small is None or len(cone_small) == 0:
        return None

    drop_result = _gaia_drop_central_star(cone_small, center)
    if drop_result is None:
        return None
    field_cone, central_cone_row, central_sep = drop_result

    field_joined = _gaia_fetch_ap_and_join(field_cone)
    if field_joined is None:
        return None

    # ----------------------------
    # 4) LOG central + field
    # ----------------------------
    cols_center = ["source_id", "ra", "dec", "phot_g_mean_mag"]
    logging.info("Gaia background search: CENTRAL (cone-only): %s", {c: central_cone_row[c] for c in cols_center} | {"sep_arcsec": central_sep})

    n_field = len(field_joined)
    g_mag_col = "phot_g_mean_mag"
    if n_field > 0 and g_mag_col in field_joined.colnames:
        g_vals = np.ma.filled(np.asarray(field_joined[g_mag_col], dtype=float), np.nan)
        g_min, g_max = np.nanmin(g_vals), np.nanmax(g_vals)
        if np.isfinite(g_min) and np.isfinite(g_max):
            logging.info("Gaia background search: FIELD stars count=%d, G_mag min=%.2f max=%.2f", n_field, float(g_min), float(g_max))
        else:
            logging.info("Gaia background search: FIELD stars count=%d", n_field)
    else:
        logging.info("Gaia background search: FIELD stars count=%d", n_field)

    print(f"Gaia background search: DONE remaining background stars={n_field}")

    return field_joined


def _gaia_cone_search(center: SkyCoord, radius_arcsec: float, g_mag_limit: float | None = None) -> Table | None:
    """Cone search on gaia_source, slice columns. If g_mag_limit is set, keep only rows with phot_g_mean_mag < g_mag_limit. Returns table or None."""
    try:
        cone_result = Gaia.cone_search(center, radius=radius_arcsec * u.arcsec)
        cone = cone_result.get_results() if hasattr(cone_result, "get_results") else cone_result
    except Exception as e:
        msg = f"Gaia background search: cone search failed: {e}"
        logging.exception(msg)
        print(msg)
        return None

    logging.info("Gaia background search: after cone search rows=%d", len(cone))

    if cone is None or len(cone) == 0:
        logging.info("Gaia background search: no sources found")
        return None

    cone_small = cone[["source_id", "ra", "dec", "parallax", "phot_g_mean_mag"]]

    if g_mag_limit is not None:
        print(f"Gaia background search: applying G<{g_mag_limit} filter")
        logging.info("Gaia background search: applying G<%s filter", g_mag_limit)
        cone_small = cone_small[cone_small["phot_g_mean_mag"] < g_mag_limit]
        logging.info("Gaia background search: after mag filter rows=%d", len(cone_small))
    else:
        logging.info("Gaia background search: no magnitude filter (g_mag_limit=None)")

    if len(cone_small) == 0:
        logging.info("Gaia background search: no sources after magnitude filter")
        return None

    return cone_small


def _gaia_drop_central_star(cone_small: Table, center: SkyCoord) -> tuple[Table, object, float] | None:
    """Identify central (nearest) star, remove it from table. Returns (field_cone, central_cone_row, central_sep) or None if no field left."""

    logging.info("Gaia background search: identifying central star (nearest)")

    cone_coords = SkyCoord(ra=cone_small["ra"], dec=cone_small["dec"], unit=u.deg, frame="icrs")
    seps = center.separation(cone_coords)
    seps_arcsec = seps.arcsec

    idx_center = int(np.argmin(seps_arcsec))

    central_cone_row = cone_small[idx_center]
    central_sep = float(seps_arcsec[idx_center])

    mask = np.ones(len(cone_small), dtype=bool)
    mask[idx_center] = False
    field_cone = cone_small[mask]

    logging.info("Gaia background search: after central removal rows=%d", len(field_cone))

    if len(field_cone) == 0:
        logging.info("Gaia background search: only central star present after filters")
        return None

    return (field_cone, central_cone_row, central_sep)


def _gaia_fetch_ap_and_join(field_cone: Table, ap_batch_size: int = 500) -> Table | None:
    """Query astrophysical_parameters for field source_ids in batches (sync), left-join to field_cone. Returns joined table or None on TAP/network error."""
    logging.info("Gaia background search: before AP search (remaining IDs=%d)", len(field_cone))

    ids = [int(x) for x in field_cone["source_id"]]
    ap_tables: list[Table] = []

    for start in range(0, len(ids), ap_batch_size):
        chunk = ids[start : start + ap_batch_size]
        in_list = ",".join(str(x) for x in chunk)
        ap_query = f"""
            SELECT
                source_id,
                teff_gspphot,
                radius_gspphot,
                mass_flame,
                mh_gspphot,
                logg_gspphot,
                distance_gspphot
            FROM gaiadr3.astrophysical_parameters
            WHERE source_id IN ({in_list})
        """
        try:
            job = Gaia.launch_job(ap_query)
            tbl = job.get_results() if hasattr(job, "get_results") else job
            if tbl is not None and len(tbl) > 0:
                ap_tables.append(tbl)
        except Exception as e:
            msg = f"Gaia background search: AP query failed (chunk {start}-{start + len(chunk)}): {e}"
            logging.exception(msg)
            print(msg)
            return None

    if not ap_tables:
        logging.info("Gaia background search: after AP search rows=0 (no AP data)")
        return field_cone

    ap_tbl = table_vstack(ap_tables)
    ap_rows = len(ap_tbl)
    logging.info("Gaia background search: after AP search rows=%d", ap_rows)

    field_joined = join(field_cone, ap_tbl, keys="source_id", join_type="left")
    return field_joined
