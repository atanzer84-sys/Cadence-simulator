import numpy as np
import logging
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import join, vstack as table_vstack
from domain.star import Star
from astropy.table import Table
from configs.global_config import GlobalConfig

GAIA_PROPERTIES = (
    "source_id",
    "right_ascension",
    "declination",
    "parallax",
    "gaia_magnitude",
    "effective_temperature",
    "distance",
    "radius",
    "mass",
    "metallicity",
    "surface_gravity",
)

GAIA_PROVIDES = set(GAIA_PROPERTIES)

def lookup_target_star_gaia(star_params: dict, missing_stellar_keys, cfg: GlobalConfig) -> dict:

    target_name = star_params["name"]
    logging.info("Gaia lookup required for star: %s for missing required parameters: %s", target_name, missing_stellar_keys)
    print("Gaia lookup required for star: %s" % target_name)

    try:
        right_ascension = star_params.get("right_ascension")
        declination = star_params.get("declination")

        source_id = _get_source_id(target_name, right_ascension, declination, cfg.GAIA_USE_ASYNC_JOBS)
        if source_id is None:
            raise ValueError(f"Gaia lookup failed for {target_name}: no RA/Dec available and no Gaia source_id could be resolved from SIMBAD")

        gaia_row = query_gaia_target_star(source_id, cfg.GAIA_USE_ASYNC_JOBS)

        # map Gaia columns to internal keys
        gaia_star_params = get_gaia_stellar_properties(gaia_row)

        gaia_star_params = apply_distance_from_parallax_if_missing(gaia_star_params)

        # return missing keys; some Gaia keys are always merged for downstream use even if not required in Excel
        still_missing = [k for k in missing_stellar_keys if gaia_star_params.get(k) is None]

        # need to check if some required properties are still missing.
        if still_missing:
            msg = f"Gaia lookup for {target_name} did not return requested missing keys: {still_missing}"
            logging.error(msg)
            print(msg)
            raise RuntimeError(msg)

        logging.info("Gaia parameters to merge: %s", gaia_star_params)

        # i want all properties that i fetched from gaia.
        return gaia_star_params

    except Exception as e:
        logging.error("Gaia lookup failed for %s: %s", target_name, str(e))
        print(f"Gaia lookup failed for {target_name}: {e}")
        raise

def _get_source_id(target_name: str, right_ascension: float | None, declination: float | None, GAIA_USE_ASYNC_JOBS) -> int | None:

    # use Simbad to get Gaia Source ID for the target's name
    source_id = _resolve_gaia_source_id_from_name(target_name)

    if source_id is None:
        if right_ascension is None or declination is None:
            return None
        source_id = _resolve_source_id_from_coordinates(target_name, float(right_ascension), float(declination), GAIA_USE_ASYNC_JOBS)

    return source_id

def _resolve_gaia_source_id_from_name(target_name: str) -> int | None:
    from astroquery.simbad import Simbad
    import re

    logging.info("SIMBAD lookup: resolving Gaia source_id for target_name=%s", target_name)

    simbad = Simbad()
    simbad.add_votable_fields("ids")

    try:
        result = simbad.query_object(target_name)
    except Exception as e:
        logging.exception("SIMBAD lookup failed for target_name=%s: %s", target_name, str(e))
        return None

    if result is None or len(result) == 0:
        logging.warning("SIMBAD lookup returned no result for target_name=%s", target_name)
        return None

    ids_value = result["ids"][0]
    if ids_value is None:
        logging.warning("SIMBAD lookup returned no IDS field for target_name=%s", target_name)
        return None

    ids_text = str(ids_value)
    logging.info("SIMBAD IDS for %s: %s", target_name, ids_text)

    patterns = [
        r"Gaia DR3 (\d+)",
        r"Gaia EDR3 (\d+)",
        r"Gaia DR2 (\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, ids_text)
        if match:
            source_id = int(match.group(1))
            logging.info("SIMBAD match found for %s: pattern=%s source_id=%s", target_name, pattern, source_id)
            return source_id

    logging.warning("SIMBAD lookup: no Gaia source_id found in IDS for target_name=%s", target_name)
    return None

def _resolve_source_id_from_coordinates(target_name: str, right_ascension: float, declination: float, GAIA_USE_ASYNC_JOBS) -> int:

    target_coord = _get_target_coordinates(target_name, right_ascension, declination)
    cone_table = _gaia_cone_search(target_coord, radius_arcsec=6.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS)

    if cone_table is None or len(cone_table) == 0:
        msg = f"No Gaia cone result found for {target_name}"
        logging.error(msg)
        print(msg)
        raise RuntimeError(msg)

    source_id = _find_target_source_id(cone_table, target_coord)
    if source_id is None:
        msg = f"No Gaia central match found for {target_name}"
        logging.error(msg)
        print(msg)
        return None

    return source_id

def _get_target_coordinates(target_name: str, right_ascension: float | None, declination: float | None) -> SkyCoord:

    if right_ascension is None or declination is None:
        raise ValueError(f"Gaia lookup failed for {target_name}: no RA/Dec available and no Gaia source_id could be resolved from SIMBAD")

    target_coord = SkyCoord(ra=right_ascension * u.deg, dec=declination * u.deg, frame="icrs")

    logging.info("Gaia lookup target coords for %s: ra=%s deg dec=%s deg frame=%s", target_name, float(target_coord.ra.deg), float(target_coord.dec.deg), getattr(target_coord.frame, "name", str(target_coord.frame)))

    return target_coord

def _gaia_cone_search(center: SkyCoord, radius_arcsec: float, g_mag_limit: float, GAIA_USE_ASYNC_JOBS) -> Table | None:
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

    if cone is None or len(cone) == 0:
        logging.info("Gaia background search: no sources found")
        return None

    cone_small = cone[["source_id", "ra", "dec", "parallax", "phot_g_mean_mag"]]

    if g_mag_limit is not None:
        print(f"Gaia background search: applying G<{g_mag_limit} filter")
        cone_small = cone_small[cone_small["phot_g_mean_mag"] < g_mag_limit]
        logging.info("Gaia background search: applying filter G<%s after mag filter rows=%d", g_mag_limit, len(cone_small))
        if len(cone_small) == 0:
            logging.info("Gaia background search: no sources after magnitude filter")
            return None

    # TODO: REMOVE THIS AGAIN
    logging.info("Gaia background search: cone rows=%d ", len(cone))
    for i, row in enumerate(cone_small):
        logging.info("Gaia background search: cone_row[%d] source_id=%s ra=%s deg dec=%s deg parallax=%s phot_g_mean_mag=%s", i, int(row["source_id"]), _to_float(row["ra"]), _to_float(row["dec"]), _to_float(row["parallax"]), _to_float(row["phot_g_mean_mag"]))

    return cone_small

def _find_target_source_id(cone_result: Table, target_coord: SkyCoord) -> int | None:
    """
    Find the row in cone_small that is closest to the given center position.

    Returns (row, separation_arcsec) or (None, None) if cone_small is empty.
    """
    cone_coords = SkyCoord(ra=cone_result["ra"], dec=cone_result["dec"], unit=u.deg, frame="icrs")
    
    separations = target_coord.separation(cone_coords)
    separations_arcsec = separations.arcsec

    # TODO: REMOVE AGAIN
    for i in range(len(separations_arcsec)):
        logging.info("Gaia central-row: cone_coords[%d] ra=%s deg dec=%s deg seps_arcsec=%s source_id=%s", i, float(cone_coords.ra.deg[i]), float(cone_coords.dec.deg[i]), float(separations_arcsec[i]), int(cone_result[i]["source_id"]))

    idx_center = int(np.argmin(separations_arcsec))
    central_cone_row = cone_result[idx_center]
    central_sep = float(separations_arcsec[idx_center])
    central_source_id = int(central_cone_row["source_id"]) if "source_id" in cone_result.colnames else None
    logging.info("Gaia central-row: idx_center=%d central_source_id=%s central_sep_arcsec=%.6f", idx_center, central_source_id, central_sep)
    return central_source_id

def query_gaia_target_star(sourceID, GAIA_USE_ASYNC_JOBS):
    query = _gaia_query_for_source_id(sourceID)
    gaia_result_row = _run_gaia_query(query, GAIA_USE_ASYNC_JOBS)

    if len(gaia_result_row) == 0:
        msg = f"No Gaia row returned for source_id={sourceID}"
        logging.error(msg)
        print(msg)
        raise RuntimeError(msg)

    logging.info("Gaia row for source_id=%s: %s", sourceID, {col: gaia_result_row[0][col] for col in gaia_result_row.colnames})
    return gaia_result_row[0]

def _gaia_query_for_source_id(source_id: int) -> str:
    return f"{_gaia_select_joined_base()} WHERE gs.source_id = {int(source_id)}"

def _gaia_select_joined_base() -> str:
    return f"""
        SELECT
            gs.source_id,
            gs.ra AS right_ascension,
            gs.dec AS declination,
            gs.parallax AS parallax,
            gs.phot_g_mean_mag AS gaia_magnitude,
            COALESCE(ap.teff_gspphot, ap.teff_gspspec, supp.teff_gspspec_ann, gs.rv_template_teff) AS effective_temperature,
            COALESCE(ap.distance_gspphot, supp.distance_gspphot_phoenix, supp.distance_gspphot_marcs) AS "distance",
            COALESCE(ap.radius_gspphot, ap.radius_flame, supp.radius_flame_spec, supp.radius_gspphot_a, supp.radius_gspphot_marcs, supp.radius_gspphot_phoenix) AS radius,
            COALESCE(ap.mass_flame, supp.mass_flame_spec) AS mass,
            ap.mh_gspphot AS metallicity,
            ap.logg_gspphot AS surface_gravity
        FROM gaiadr3.gaia_source AS gs
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
        LEFT JOIN gaiadr3.astrophysical_parameters_supp AS supp ON gs.source_id = supp.source_id
    """

def _run_gaia_query(query: str, GAIA_USE_ASYNC_JOBS):
    from astroquery.gaia import Gaia

    use_async = GAIA_USE_ASYNC_JOBS
    if use_async:
        job = Gaia.launch_job_async(query)
    else:
        job = Gaia.launch_job(query)
    return job.get_results() if hasattr(job, "get_results") else job







def _gaia_query_for_source_ids(source_ids: list[int]) -> str:
    in_list = ",".join(str(int(x)) for x in source_ids)
    return f"{_gaia_select_joined_base()} WHERE gs.source_id IN ({in_list})"

def get_gaia_stellar_properties(gaia_row, log_output: bool = True):
    gaia_star_params = {}

    for key in GAIA_PROPERTIES:
        value = gaia_row[key]

        if key == "source_id":
            gaia_star_params[key] = int(value) if value is not None else None
        else:
            gaia_star_params[key] = _to_float(value)
            
    if log_output:
        logging.info("Gaia stellar parameters extracted: %s", gaia_star_params)    
    
    return gaia_star_params

def _to_float(value):
    if value is None:
        return None
    if np.ma.is_masked(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None

    if math.isnan(value):
        return None
    return value


def gaia_lookup_for_background_stars(star: Star, g_mag_limit, GAIA_USE_ASYNC_JOBS, radius_arcsec) -> Table | None:
    print(f"==== Gaia background search: START (g_mag_limit={g_mag_limit}, GAIA_USE_ASYNC_JOBS={GAIA_USE_ASYNC_JOBS} , radius_arcsec={radius_arcsec}) =====")
    logging.info("Gaia background search: START g_mag_limit=%s GAIA_USE_ASYNC_JOBS=%s radius_arcsec=%s", g_mag_limit, GAIA_USE_ASYNC_JOBS, radius_arcsec)

    # if we already have the source id from gaia, this is easy to remove from the gaia rows we get from the sql query.
    target_name = star.name
    target_gaia_source_id = star.gaia_source_id
    target_right_ascension = star.right_ascension
    target_declination = star.declination

    if star.gaia_source_id is None:
        target_gaia_source_id = _get_source_id(target_name, target_right_ascension, target_declination, GAIA_USE_ASYNC_JOBS)
        





    drop_result = _gaia_drop_central_star(cone_small, center)
    if drop_result is None:
        return None
    field_cone, _, _ = drop_result

    field_joined = _gaia_fetch_ap_and_join(field_cone, GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS)
    if field_joined is None:
        return None

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

    logging.info("Gaia background search completed: target=%s stars=%d mag_limit=%s GAIA_USE_ASYNC_JOBS=%s", star.name, len(field_joined), g_mag_limit, GAIA_USE_ASYNC_JOBS)
    return field_joined




  
def _gaia_drop_central_star(cone_small: Table, center: SkyCoord) -> tuple[Table, object, float] | None:
    """Identify central (nearest) star, remove it from table. Returns (field_cone, central_cone_row, central_sep) or None if no field left."""


    central_cone_row, central_sep = _find_target_source_id(cone_small, center)
    if central_cone_row is None:
        logging.info("Gaia background search: no central star found in cone")
        return None

    # build field table without the central row
    idx_center = int(np.where(cone_small["source_id"] == central_cone_row["source_id"])[0][0])
    mask = np.ones(len(cone_small), dtype=bool)
    mask[idx_center] = False
    field_cone = cone_small[mask]

    if len(field_cone) == 0:
        logging.info("Gaia background search: only central star present after filters")
        return None

    logging.info("Gaia background search: after central removal rows=%d", len(field_cone))
    return (field_cone, central_cone_row, central_sep)


def _gaia_fetch_ap_and_join(field_cone: Table, GAIA_USE_ASYNC_JOBS, ap_batch_size: int = 500) -> Table | None:
    """Query astrophysical_parameters for field source_ids in batches (sync), left-join to field_cone. Returns joined table or None on TAP/network error."""
    logging.info("Gaia background search: before AP search (remaining IDs=%d)", len(field_cone))

    ids = [int(x) for x in field_cone["source_id"]]
    ap_tables: list[Table] = []

    for start in range(0, len(ids), ap_batch_size):
        chunk = ids[start:start + ap_batch_size]
        ap_query = _gaia_query_for_source_ids(chunk)
        try:
            tbl = _run_gaia_query(ap_query, GAIA_USE_ASYNC_JOBS)
            if tbl is not None and len(tbl) > 0:
                ap_tables.append(tbl)
                # TODO: REMOVE
                ap_tbl_text = "\n".join(tbl.pformat(max_lines=-1, max_width=-1))
                logging.info("Gaia background search: AP chunk start=%d n_ids=%d n_rows=%d\n%s", start, len(chunk), len(tbl), ap_tbl_text)
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
    field_joined_text = "\n".join(field_joined.pformat(max_lines=-1, max_width=-1))
    logging.info("Gaia background search: field_joined after AP join rows=%d cols=%s\n%s", len(field_joined), list(field_joined.colnames), field_joined_text)
    return field_joined



def apply_distance_from_parallax_if_missing(star_params: dict) -> dict:
    """
    If distance is missing, set it from parallax in star_params
    using distance_pc = 1000 / parallax_mas when valid.
    """
    identifier = star_params.get("name") or star_params.get("source_id") or "<unknown>"
    distance = star_params.get("distance")
    parallax = star_params.get("parallax")

    if distance is not None:
        logging.info("Parallax distance fallback skipped for %s: distance already set (%r)", identifier, distance)
        return star_params

    if parallax is None or np.ma.is_masked(parallax):
        logging.info("Parallax distance fallback skipped for %s: no usable parallax in star_params (parallax=%r)", identifier, parallax)
        return star_params

    try:
        parallax = float(parallax)
    except Exception:
        logging.info("Parallax distance fallback skipped for %s: parallax not float-convertible (parallax=%r)", identifier, parallax)
        return star_params

    if not np.isfinite(parallax) or parallax <= 0.0:
        logging.info("Parallax distance fallback skipped for %s: parallax invalid (parallax=%r)", identifier, parallax)
        return star_params

    star_params["distance"] = 1000.0 / parallax
    logging.info("Parallax distance fallback applied for %s: parallax_mas=%s -> distance_pc=%s", identifier, parallax, star_params["distance"])
    return star_params