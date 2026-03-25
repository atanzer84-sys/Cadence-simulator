import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from astropy.table import Table
from loaders.run_waltzer_context import get_repo_root
from utils.helpers import resolve_path_under
from configs.global_config import GlobalConfig
from domain.star_catalog import StarCatalog
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from loaders.load_stellar_and_planetary_properties import load_excel_mapping, infer_mamajek, apply_log_r, get_missing_properties, apply_radius_from_teff_mag_distance_if_missing
from loaders.load_gaia import get_gaia_stellar_properties, gaia_lookup_for_background_stars, apply_distance_from_parallax_if_missing
import astropy.units as u
from astropy.coordinates import SkyCoord
from configs.global_config import get_global_config


def lookup_background_stars(nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None, ctx: RunContext, star: Star):
    print("\n==== STARTING BACKGROUND STAR LOOKUP VIA GAIA OR CSV =====")

    cfg = get_global_config()

    # TODO: IF YOU WANT TO LOAD A CATALOG OF ALL BACKGROUND STARS AND LOOK THEM UP, THIS WOULD BE THE PLACE.
    # lookup star_name.csv and load it, so we do not have to query gaia
    table = _load_or_query_background_star_table(ctx, star, cfg)

    if table is None or len(table) == 0:
        return StarCatalog()

    table = _apply_background_star_magnitude_cutoff(table, cfg.magnitude_cutoff)

    if len(table) == 0:
        return StarCatalog()

    table = _annotate_background_star_offsets_arcsec(table, star)
    table = _drop_stars_outside_max_radius(table, nuv, vis, nir)
    catalog = create_background_star_catalog(table, cfg)

    logging.info("Background star catalog prepared: stars=%d", len(catalog.stars_by_id))

    return catalog

def _load_or_query_background_star_table(ctx: RunContext, star: Star, cfg: GlobalConfig) -> Table | None:
    table = _load_background_csv_if_exists(star)

    if table is None:
        table = gaia_lookup_for_background_stars(star, g_mag_limit=cfg.magnitude_cutoff, GAIA_USE_ASYNC_JOBS=cfg. GAIA_USE_ASYNC_JOBS, radius_arcsec=cfg.gaia_conesearch_radius_arcsec)

        if table is not None and len(table) > 0:
            _save_background_stars_csv(table, ctx.output_dir, star.name)

    return table

def _load_background_csv_if_exists(star: Star, repo_root=None) -> Table | None:
    repo_root = get_repo_root() if repo_root is None else repo_root
    csv_name = star.name.replace(" ", "_")
    csv_path = resolve_path_under(repo_root, "data", "BackgroundStars", f"{csv_name}.csv")
    logging.info("Background stars: loading cached CSV: %s", csv_path)

    if not csv_path.exists():
        return None

    table = Table.read(csv_path, format="csv")
    required_columns = {"right_ascension", "declination"}
    missing_columns = sorted(required_columns - set(table.colnames))
    if missing_columns:
        raise ValueError(
            f"Background CSV missing required columns {missing_columns}: {csv_path}"
        )

    logging.info("Loaded background CSV for %s: rows=%d, columns=%s", star.name, len(table), list(table.colnames))
    return table

def _save_background_stars_csv(table: Table, output_dir, star_name: str) -> None:
    """Write background stars table to CSV in output_dir. Use same name as cache so it can be moved to data/BackgroundStars/."""
    csv_name = star_name.replace(" ", "_")
    csv_path = output_dir / f"{csv_name}.csv"
    table.write(csv_path, format="ascii.csv", overwrite=True)
    msg = f"Background stars: saved to {csv_path} (move to data/BackgroundStars/ for cache)"
    logging.info(msg)
    print(msg)

def _apply_background_star_magnitude_cutoff(table: Table, max_mag: float) -> Table:
    if "gaia_magnitude" not in table.colnames:
        logging.info("Background star magnitude filter skipped: 'gaia_magnitude' column not present")
        return table

    table = table[table["gaia_magnitude"] <= max_mag]
    logging.info("Background star magnitude filter applied: max_mag=%s remaining=%d", max_mag, len(table))
    return table

def _annotate_background_star_offsets_arcsec(table: Table, target_star: Star) -> Table:
    ra0 = float(target_star.right_ascension)
    dec0 = float(target_star.declination)
    target = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg, frame="icrs")

    sources_coord = SkyCoord(
        ra=np.asarray(table["right_ascension"], dtype=float) * u.deg,
        dec=np.asarray(table["declination"], dtype=float) * u.deg,
        frame="icrs",
    )

    dlon, dlat = target.spherical_offsets_to(sources_coord)

    use_cached_sep = False
    if "sep_arcsec" in table.colnames:
        try:
            sep_values = np.asarray(table["sep_arcsec"], dtype=float)
            if len(sep_values) == len(table) and np.all(np.isfinite(sep_values)):
                use_cached_sep = True
        except Exception:
            use_cached_sep = False

    if not use_cached_sep:
        sep_values = target.separation(sources_coord).to(u.arcsec).value

    table = table.copy()
    table["relative_dx_arcsec"] = dlon.to(u.arcsec).value
    table["relative_dy_arcsec"] = dlat.to(u.arcsec).value
    table["separation_arcsec"] = sep_values

    logging.info("Background star offsets computed: stars=%d used_cached_sep=%s", len(table), use_cached_sep)

    return table

def _drop_stars_outside_max_radius(table: Table, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None) -> Table:
    """Filter Gaia background stars to those that can reach any active channel."""

    radii_arcsec = []

    if nuv is not None:
        radii_arcsec.append(_spectroscopy_radius_arcsec(nuv))

    if vis is not None:
        radii_arcsec.append(_spectroscopy_radius_arcsec(vis))

    if nir is not None:
        radii_arcsec.append(_photometry_radius_arcsec(nir))

    if not radii_arcsec:
        logging.info("Background star radius filter skipped: no channels enabled; input_rows=%d",
            len(table))
        return table

    max_radius_arcsec = max(radii_arcsec)

    if max_radius_arcsec <= 0.0:
        logging.info("Background star radius filter no-op: non-positive max_radius_arcsec=%.6f; input_rows=%d", max_radius_arcsec, len(table))
        return table

    before = len(table)
    result = table[table["separation_arcsec"] <= max_radius_arcsec]
    after = len(result)

    logging.info("Background star reach radii: radii_arcsec=%s max_radius_arcsec=%.6f kept=%d/%d", [round(r, 6) for r in radii_arcsec], max_radius_arcsec, after, before)

    return result

def _spectroscopy_radius_arcsec(channel: SpectroscopyChannel) -> float:
    x = float(channel.slit_half_width_arcsec)
    y = float(channel.slit_half_length_arcsec)
    return (x * x + y * y) ** 0.5

def _photometry_radius_arcsec(channel: PhotometryChannel) -> float:
    half_width_arcsec = 0.5 * float(channel.x_pixels) * float(channel.pixel_scale)
    half_height_arcsec = 0.5 * float(channel.y_pixels) * float(channel.pixel_scale)
    return (half_width_arcsec * half_width_arcsec + half_height_arcsec * half_height_arcsec) ** 0.5

def create_background_star_catalog(table: Table, cfg: GlobalConfig):
    catalog = StarCatalog()
    required_keys = load_required_stellar_parameters()

    for row in table:
        star_params = get_gaia_stellar_properties(row, log_output=False)

        _set_background_star_name(star_params, row)
        star_params = apply_distance_from_parallax_if_missing(star_params)
        star_params = apply_radius_from_teff_mag_distance_if_missing(star_params)

        if star_params.get("effective_temperature") is not None:
            star_params = infer_mamajek(star_params, log_output=False)
        
        if star_params.get("radius") is not None:
            star_params = apply_log_r(star_params, cfg, log_output=False)

        if not _ensure_required_properties(star_params, required_keys):
            continue

        bg_star = Star.from_params(star_params, required_keys, log_output = False)

        catalog.add_star(bg_star.name, bg_star)

        dx = float(row["relative_dx_arcsec"])
        dy = float(row["relative_dy_arcsec"])
        catalog.set_offset_arcsec(bg_star.name, dx, dy)


        formatted_id = f"{int(bg_star.name.split('_')[1]):,}".replace(",", " ")
        mass_str = f"{bg_star.mass:.3f}" if bg_star.mass is not None else "n/a"
        logging.info("Background star added: star_id_formatted=%s star_id=%s mag=%.3f teff=%.0f radius=%.3f mass=%s ra=%.6f dec=%.6f dist=%.2f dx=%.3f dy=%.3f", formatted_id, bg_star.name, bg_star.gaia_magnitude, bg_star.effective_temperature, bg_star.radius, mass_str, bg_star.right_ascension, bg_star.declination, bg_star.distance_pc, dx, dy)
    

    return catalog

def load_required_stellar_parameters():
    mapping = load_excel_mapping()
    return mapping["required_stellar_parameters"]


def _set_background_star_name(star_params: dict, row) -> None:
    """Set star_params['name'] from row source_id, or '0000' if missing."""
    if "source_id" in row.colnames and row["source_id"] is not None:
        star_params["name"] = f"gaia_{int(row['source_id'])}"
    else:
        star_params["name"] = "0000"

def _ensure_required_properties(star_params: dict, required_keys: list[str]) -> bool:
    """Return False if any required keys are missing (logs and skip). Otherwise return True."""
    missing = get_missing_properties(star_params, required_keys, log_output=False)
    if missing:
        logging.info("Background star %s skipped. Missing: %s", star_params.get("name"), missing)
        return False
    return True






