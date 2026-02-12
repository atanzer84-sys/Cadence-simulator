import numpy as np
import logging
import math

def lookup_star_gaia(star_params: dict, missing_star) -> dict:
    
    target_name = star_params["name"]
    logging.info("Gaia lookup required for star: %s for missing required parameters: %s", target_name, missing_star)
    print("Gaia lookup required for star: %s" % target_name)

    try:

        gaia_result = query_gaia_by_name(target_name)
        source_id = select_source_id_from_best_gaia_match(gaia_result)
        print("source id:", source_id)

        if source_id is None:
            logging.warning("No Gaia match found for %s", target_name)
            return {}

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


def query_gaia_by_name(target_name, radius_arcsec=2.0):
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord.from_name(target_name)
    
    job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
    results = job.get_results()
    return results

def select_source_id_from_best_gaia_match(gaia_table):
    if len(gaia_table) == 0:
        return None

    sourceID = gaia_table[np.argmin(gaia_table["phot_g_mean_mag"])]
    return int(sourceID["source_id"])

def query_gaia(sourceID):
    from astroquery.gaia import Gaia
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
    query_job = Gaia.launch_job_async(query)
    gaia_result_row = query_job.get_results()

    if len(gaia_result_row) == 0:
        logging.warning("No Gaia joined row returned for (source_id=%s)", sourceID)
        return {}
    
    logging.info("Gaia row for source_id=%s: %s", sourceID, {col: gaia_result_row[0][col] for col in gaia_result_row.colnames})

    return gaia_result_row[0]

def get_gaia_stellar_properties(gaia_row):
    def _to_float(value):
        if value is None:
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
        "log_r": None
    }
    logging.info("Gaia stellar parameters extracted: %s", gaia_star_params)
    return gaia_star_params