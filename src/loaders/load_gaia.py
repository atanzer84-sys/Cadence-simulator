import numpy as np
import logging
import math
from astropy.coordinates import SkyCoord
from astropy.table import join, vstack as table_vstack
import astropy.units as u
from domain.star import Star
from astropy.table import Table

GAIA_PROVIDES = {
    "right_ascension",
    "declination",
    "gaia_magnitude",
    "v_magnitude",
    "effective_temperature",
    "radius",
    "mass",
    "metallicity",
    "surface_gravity",
    "distance",
}


GAIA_USE_ASYNC_JOBS: bool = True


def lookup_star_gaia(star_params: dict, missing_star) -> dict:
    target_name = star_params["name"]
    logging.info("Gaia lookup required for star: %s for missing required parameters: %s", target_name, missing_star)
    print("Gaia lookup required for star: %s" % target_name)

    try:
        # Decide how to get the sky position: prefer provided RA/Dec, fall back to name resolution.
        ra = star_params.get("right_ascension")
        dec = star_params.get("declination")

        if ra is not None and dec is not None:
            logging.info("Gaia lookup using provided RA/Dec for star %s: ra=%s deg, dec=%s deg (missing=%s)", target_name, ra, dec, missing_star)
            center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        else:
            logging.info("Gaia lookup using name resolution for star %s (no RA/Dec in parameters; missing=%s)", target_name, missing_star)
            center = SkyCoord.from_name(target_name)

        cone_table = _gaia_cone_search(center, radius_arcsec=2.0, g_mag_limit=None)
        if cone_table is None or len(cone_table) == 0:
            logging.warning("No Gaia cone result found for %s", target_name)
            return {}

        central_row, central_sep = _find_central_row(cone_table, center)
        if central_row is None:
            logging.warning("No Gaia central match found for %s", target_name)
            return {}

        source_id = int(central_row["source_id"])
        print("source id (nearest):", source_id, "| sep_arcsec=", central_sep)

        gaia_row = query_gaia(source_id)
        if not gaia_row:
            return {}

        # map Gaia columns to internal keys
        gaia_star_params = get_gaia_stellar_properties(gaia_row)

        # return only missing keys
        gaia_filtered = {k: gaia_star_params.get(k) for k in missing_star if k in gaia_star_params}

        logging.info("Gaia parameters to merge: %s", gaia_filtered)

        return gaia_filtered

    except Exception as e:
        logging.error("Gaia lookup failed for %s: %s", target_name, str(e))
        return {}

def query_gaia(sourceID):
    query = f"""
        SELECT
            gs.source_id,
            gs.ra,
            gs.dec,
            gs.phot_g_mean_mag,

            ap.teff_gspphot,
            ap.radius_gspphot,
            ap.mass_flame,
            ap.mh_gspphot,
            ap.logg_gspphot,
            ap.distance_gspphot

        FROM gaiadr3.gaia_source AS gs
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
            ON gs.source_id = ap.source_id

        WHERE gs.source_id = {sourceID}
        """
    gaia_result_row = _run_gaia_job(query)

    if len(gaia_result_row) == 0:
        logging.warning("No Gaia joined row returned for (source_id=%s)", sourceID)
        return {}
    
    logging.info("Gaia row for source_id=%s: %s", sourceID, {col: gaia_result_row[0][col] for col in gaia_result_row.colnames})

    return gaia_result_row[0]

def get_gaia_stellar_properties(gaia_row, log_output: bool = True):
    def _to_float(value):
        if value is None:
            return None
        if np.ma.is_masked(value):
            return None
        value = float(value)
        return None if math.isnan(value) else value
    
    gaia_star_params = {
        "effective_temperature": _to_float(gaia_row.get("teff_gspphot")),
        "radius": _to_float(gaia_row.get("radius_gspphot")),
        "mass": _to_float(gaia_row.get("mass_flame")),
        "metallicity": _to_float(gaia_row.get("mh_gspphot")),
        "surface_gravity": _to_float(gaia_row.get("logg_gspphot")),
        "right_ascension": _to_float(gaia_row.get("ra")),
        "declination": _to_float(gaia_row.get("dec")),
        "distance": _to_float(gaia_row.get("distance_gspphot")),
        "v_magnitude": _to_float(gaia_row.get("phot_g_mean_mag")),
        "gaia_magnitude": _to_float(gaia_row.get("phot_g_mean_mag")),
    }
    if log_output:
        logging.info("Gaia stellar parameters extracted: %s", gaia_star_params)    
    
    return gaia_star_params





def gaia_lookup_for_background_stars(star: Star, g_mag_limit) -> Table | None:
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

    print(f"==== Gaia background search: START (g_mag_limit={g_mag_limit}) =====")
    logging.info("Gaia background search: START (g_mag_limit=%s)", g_mag_limit)

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
    logging.info("Gaia background search: DONE remaining background stars=%d", n_field)

    return field_joined


def _run_gaia_job(query: str, async_override: bool | None = None):
    """
    Run a Gaia TAP job with a shared sync/async switch.

    Parameters
    ----------
    query : str
        ADQL query string.
    async_override : bool | None
        If True, force async; if False, force sync; if None, use GAIA_USE_ASYNC_JOBS.
    """
    from astroquery.gaia import Gaia

    use_async = GAIA_USE_ASYNC_JOBS if async_override is None else async_override
    if use_async:
        job = Gaia.launch_job_async(query)
    else:
        job = Gaia.launch_job(query)
    return job.get_results() if hasattr(job, "get_results") else job


def _gaia_cone_search(center: SkyCoord, radius_arcsec: float, g_mag_limit: float | None = None) -> Table | None:
    """Cone search on gaia_source, slice columns. If g_mag_limit is set, keep only rows with phot_g_mean_mag < g_mag_limit. Returns table or None."""
    try:
        from astroquery.gaia import Gaia

        use_async = GAIA_USE_ASYNC_JOBS
        if use_async:
            cone_result = Gaia.cone_search_async(center, radius=radius_arcsec * u.arcsec)
        else:
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

    
def _find_central_row(cone_small: Table, center: SkyCoord) -> tuple[object | None, float | None]:
    """
    Find the row in cone_small that is closest to the given center position.

    Returns (row, separation_arcsec) or (None, None) if cone_small is empty.
    """
    if cone_small is None or len(cone_small) == 0:
        return None, None

    cone_coords = SkyCoord(ra=cone_small["ra"], dec=cone_small["dec"], unit=u.deg, frame="icrs")
    seps = center.separation(cone_coords)
    seps_arcsec = seps.arcsec

    idx_center = int(np.argmin(seps_arcsec))
    central_cone_row = cone_small[idx_center]
    central_sep = float(seps_arcsec[idx_center])
    return central_cone_row, central_sep


def _gaia_drop_central_star(cone_small: Table, center: SkyCoord) -> tuple[Table, object, float] | None:
    """Identify central (nearest) star, remove it from table. Returns (field_cone, central_cone_row, central_sep) or None if no field left."""

    logging.info("Gaia background search: identifying central star (nearest)")

    central_cone_row, central_sep = _find_central_row(cone_small, center)
    if central_cone_row is None:
        logging.info("Gaia background search: no central star found in cone")
        return None

    # build field table without the central row
    idx_center = int(
        np.where(cone_small["source_id"] == central_cone_row["source_id"])[0][0]
    )
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
            tbl = _run_gaia_job(ap_query, async_override=False)
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
