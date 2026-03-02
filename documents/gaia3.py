#!/usr/bin/env python3
"""
VERIFICATION SCRIPT: Unified ADQL vs Manual Two-Step
Run this to confirm the results are identical to your previous method.
"""

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import numpy as np

def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)

def gaia_unified_verification(target_name: str, radius_arcsec: float = 150.0):
    ts(f"Step 1: Resolving {target_name}...")
    coord = SkyCoord.from_name(target_name)
    ra_deg = coord.ra.deg
    dec_deg = coord.dec.deg
    radius_deg = radius_arcsec / 3600.0

    # This is the "Magic" Query. 
    # It does the CONE SEARCH and the JOIN in one database operation.
    ts(f"Step 2: Running Unified ADQL (Cone + Join)...")
    
    adql = f"""
    SELECT 
        gs.source_id, gs.ra, gs.dec, gs.parallax, gs.phot_g_mean_mag,
        ap.teff_gspphot, ap.radius_gspphot, ap.mass_flame, 
        ap.mh_gspphot, ap.logg_gspphot, ap.distance_gspphot
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
    WHERE CONTAINS(
        POINT('ICRS', gs.ra, gs.dec),
        CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
    ) = 1
    """

    job = Gaia.launch_job_async(adql)
    results = job.get_results()
    
    ts(f"Step 3: Filtering results locally to match your G < 16 rule...")
    # Matches your: cone_small = cone_small[cone_small["phot_g_mean_mag"] < 16.0]
    final_table = results[results["phot_g_mean_mag"] < 16.0]
    
    ts(f"Rows found: {len(final_table)}")
    final_table.sort("phot_g_mean_mag")
    
    return final_table, coord

def main():
    TARGET_NAME = "HD 2685"
    RADIUS_ARCSEC = 150.0
    tbl, coord = gaia_unified_verification(TARGET_NAME, radius_arcsec=RADIUS_ARCSEC)

    cols = [
        "source_id", "ra", "dec", "parallax", "phot_g_mean_mag",
        "teff_gspphot", "radius_gspphot", "mass_flame",
        "mh_gspphot", "logg_gspphot", "distance_gspphot",
    ]
    
    print("\n" + "\t".join(cols))
    for row in tbl:
        # Handling masked values just like your output
        print("\t".join(str(row[c]) if not np.ma.is_masked(row[c]) else "None" for c in cols))

if __name__ == "__main__":
    main()