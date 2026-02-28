#!/usr/bin/env python3
"""
Gaia DR3: print ALL sources within 150 arcsec of a target name,
including selected gaia_source + astrophysical_parameters columns.
"""

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime

from astropy.table import join

def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)

def gaia_nearby_stars_with_params_by_name(target_name: str, radius_arcsec: float = 150.0):
    ts("Resolving target name...")

    coord = SkyCoord.from_name(target_name)

    # 1) fast: cone search in gaia_source
    ts("Running cone search on gaia_source (150 arcsec)...")

    job1 = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
    cone = job1.get_results()
    if len(cone) == 0:
        return cone, coord
    ts(f"Cone search done. Rows: {len(cone)}")

    # keep only what you need from gaia_source
    cone_small = cone["source_id", "ra", "dec", "parallax", "phot_g_mean_mag"]
    ts("Filtering G < 15...")
    cone_small = cone_small[cone_small["phot_g_mean_mag"] < 16.0]
    ts("Rows after mag filter:", len(cone_small))
    # 2) pull only AP columns for these IDs (no spatial function)
    ids = [str(int(x)) for x in cone_small["source_id"]]
    ts("Querying astrophysical_parameters for", len(ids), "source_ids...")
    ids_sql = ",".join(ids)

    query = f"""
    SELECT
        source_id,
        teff_gspphot,
        radius_gspphot,
        mass_flame,
        mh_gspphot,
        logg_gspphot,
        distance_gspphot
    FROM gaiadr3.astrophysical_parameters
    WHERE source_id IN ({ids_sql})
    """

    job2 = Gaia.launch_job_async(query)
    ap = job2.get_results()
    ts("AP query returned rows:", len(ap))
    # 3) local join on source_id
    out = join(cone_small, ap, keys="source_id", join_type="left")

    ts("Final usable stars:", len(out))
    # optional: keep same ordering as before
    out.sort("phot_g_mean_mag")
    return out, coord


def main():
    TARGET_NAME = "KELT-9"
    RADIUS_ARCSEC = 150.0

    tbl, coord = gaia_nearby_stars_with_params_by_name(TARGET_NAME, radius_arcsec=RADIUS_ARCSEC)

    ts("Target:", TARGET_NAME)
    ts("Resolved coord (ICRS):", coord.to_string("hmsdms"))
    ts("Radius (arcsec):", RADIUS_ARCSEC)
    ts("Rows:", len(tbl))
    ts()

    cols = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "phot_g_mean_mag",
        "teff_gspphot",
        "radius_gspphot",
        "mass_flame",
        "mh_gspphot",
        "logg_gspphot",
        "distance_gspphot",
    ]
    print("\t".join(cols))
    for row in tbl:
        print("\t".join(str(row[c]) for c in cols))


if __name__ == "__main__":
    main()