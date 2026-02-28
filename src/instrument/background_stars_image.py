import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
import logging
import numpy as np
import math

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import join
from astroquery.gaia import Gaia

def generate_Background_Stars_Image(channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    pass

# def get_background_candidates(star: Star):

#     radius_arcsec = 150.0
#     g_mag_limit = 16.0

#     print("Gaia background search: starting cone query")
#     # 1) single Gaia query
#     tbl = query_gaia_cone(
#         star.right_ascension,
#         star.declination,
#         radius_arcsec=radius_arcsec,
#     )
#     print(f"Gaia background search: cone query returned {len(tbl)} rows")
#     if tbl is None or len(tbl) == 0:
#         logging.info("Gaia background candidates: none found (radius=%s arcsec)", radius_arcsec)
#         return None

#     print("Gaia background search: applying magnitude filter")
#     # 2) optional brightness filter (in-memory)
#     if "phot_g_mean_mag" in tbl.colnames:
#         tbl = tbl[tbl["phot_g_mean_mag"] < g_mag_limit]

#     if len(tbl) == 0:
#         logging.info("Gaia background candidates: none after G<%s filter", g_mag_limit)
#         return None

#     # 3) remove central star (nearest by separation, in-memory)
#     center = SkyCoord(ra=star.right_ascension * u.deg, dec=star.declination * u.deg, frame="icrs")
#     coords = SkyCoord(ra=tbl["ra"] * u.deg, dec=tbl["dec"] * u.deg, frame="icrs")
#     seps_arcsec = center.separation(coords).arcsec

#     idx_center = int(seps_arcsec.argmin())
#     removed_source_id = tbl["source_id"][idx_center]
#     removed_sep = float(seps_arcsec[idx_center])

#     keep = [True] * len(tbl)
#     keep[idx_center] = False
#     tbl2 = tbl[keep]

#     logging.info(
#         "Gaia background candidates: removed central source_id=%s (sep_arcsec=%s). Remaining=%d",
#         removed_source_id, removed_sep, len(tbl2)
#     )

#     if len(tbl2) == 0:
#         logging.info("Gaia background candidates: only central star present")
#         return None

#     # 4) log “table” rows (compact)
#     cols = [
#         "source_id", "ra", "dec", "phot_g_mean_mag",
#         "teff_gspphot", "radius_gspphot", "mass_flame",
#         "mh_gspphot", "logg_gspphot", "distance_gspphot",
#     ]
#     cols = [c for c in cols if c in tbl2.colnames]

#     logging.info("Gaia background candidates rows (count=%d):", len(tbl2))
#     for row in tbl2:
#         logging.info("  %s", {c: row[c] for c in cols})

#     return tbl2





def get_background_candidates(star: Star):


    radius_arcsec = 150.0
    g_mag_limit = 16.0

    print("Gaia background search: START")
    logging.info("Gaia background search: START")

    # ----------------------------
    # 1) fast cone search (gaia_source via cone_search)
    # ----------------------------
    print("Gaia background search: before cone search")
    logging.info("Gaia background search: before cone search (radius_arcsec=%s)", radius_arcsec)

    center = SkyCoord(
        ra=star.right_ascension * u.deg,
        dec=star.declination * u.deg,
        frame="icrs",
    )

    job1 = Gaia.cone_search_async(center, radius=radius_arcsec * u.arcsec)
    cone = job1.get_results()

    print(f"Gaia background search: after cone search rows={len(cone)}")
    logging.info("Gaia background search: after cone search rows=%d", len(cone))

    if cone is None or len(cone) == 0:
        logging.info("Gaia background search: no sources found")
        return None

    # exact gaia2 style: slice exact columns + filter
    cone_small = cone[["source_id", "ra", "dec", "parallax", "phot_g_mean_mag"]]

    print(f"Gaia background search: applying G<{g_mag_limit} filter")
    logging.info("Gaia background search: applying G<%s filter", g_mag_limit)

    cone_small = cone_small[cone_small["phot_g_mean_mag"] < g_mag_limit]

    print(f"Gaia background search: after mag filter rows={len(cone_small)}")
    logging.info("Gaia background search: after mag filter rows=%d", len(cone_small))

    if len(cone_small) == 0:
        logging.info("Gaia background search: no sources after magnitude filter")
        return None

    # ----------------------------
    # 2) remove central star (nearest by separation)
    # numerical separation (no SkyCoord array)
    # ----------------------------
    print("Gaia background search: identifying central star (nearest)")
    logging.info("Gaia background search: identifying central star (nearest)")

    ra0 = float(star.right_ascension)
    dec0 = float(star.declination)

    ra0_rad = math.radians(ra0)
    dec0_rad = math.radians(dec0)

    # compute separations to all cone rows
    seps_arcsec = np.zeros(len(cone_small), dtype=np.float64)

    for i in range(len(cone_small)):
        ra = float(cone_small["ra"][i])
        dec = float(cone_small["dec"][i])

        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)

        # spherical law of cosines
        cos_d = math.sin(dec0_rad) * math.sin(dec_rad) + math.cos(dec0_rad) * math.cos(dec_rad) * math.cos(ra_rad - ra0_rad)

        # clamp numeric noise
        if cos_d > 1.0:
            cos_d = 1.0
        if cos_d < -1.0:
            cos_d = -1.0

        d_rad = math.acos(cos_d)
        d_arcsec = d_rad * (180.0 / math.pi) * 3600.0
        seps_arcsec[i] = d_arcsec

    idx_center = int(np.argmin(seps_arcsec))

    central_cone_row = cone_small[idx_center]
    central_source_id = int(central_cone_row["source_id"])
    central_sep = float(seps_arcsec[idx_center])

    print(f"Gaia background search: CENTRAL source_id={central_source_id} sep_arcsec={central_sep}")
    logging.info(
        "Gaia background search: CENTRAL source_id=%s sep_arcsec=%s Gmag=%s ra=%s dec=%s",
        central_source_id,
        central_sep,
        float(central_cone_row["phot_g_mean_mag"]),
        float(central_cone_row["ra"]),
        float(central_cone_row["dec"]),
    )

    # remove central from cone_small
    mask = np.ones(len(cone_small), dtype=bool)
    mask[idx_center] = False
    field_cone = cone_small[mask]

    print(f"Gaia background search: after central removal rows={len(field_cone)}")
    logging.info("Gaia background search: after central removal rows=%d", len(field_cone))

    if len(field_cone) == 0:
        logging.info("Gaia background search: only central star present after filters")
        return None

    # ----------------------------
    # 3) AP query for remaining IDs only (gaia2 speed trick)
    # ----------------------------
    print("Gaia background search: before AP search (remaining IDs)")
    logging.info("Gaia background search: before AP search (remaining IDs=%d)", len(field_cone))

    ids = [int(x) for x in field_cone["source_id"]]
    in_list = ",".join(str(x) for x in ids)

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

    ap_tbl = Gaia.launch_job_async(ap_query).get_results()

    ap_rows = 0 if ap_tbl is None else len(ap_tbl)
    print(f"Gaia background search: after AP search rows={ap_rows}")
    logging.info("Gaia background search: after AP search rows=%d", ap_rows)

    # join cone basics + AP params
    if ap_tbl is None or len(ap_tbl) == 0:
        field_joined = field_cone
    else:
        field_joined = join(field_cone, ap_tbl, keys="source_id", join_type="left")

    # ----------------------------
    # 4) LOG central + field
    # ----------------------------
    cols_center = ["source_id", "ra", "dec", "phot_g_mean_mag"]
    logging.info("Gaia background search: CENTRAL (cone-only): %s", {c: central_cone_row[c] for c in cols_center} | {"sep_arcsec": central_sep})

    cols_field = [
        "source_id", "ra", "dec", "phot_g_mean_mag",
        "teff_gspphot", "radius_gspphot", "mass_flame",
        "mh_gspphot", "logg_gspphot", "distance_gspphot",
    ]
    cols_field = [c for c in cols_field if c in field_joined.colnames]

    logging.info("Gaia background search: FIELD stars (count=%d)", len(field_joined))
    for row in field_joined:
        logging.info("FIELD: %s", {c: row[c] for c in cols_field})

    print(f"Gaia background search: DONE remaining={len(field_joined)}")
    logging.info("Gaia background search: DONE remaining=%d", len(field_joined))

    return field_joined