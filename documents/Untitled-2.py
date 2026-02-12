
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

def query_gaia_by_name(target_name, radius_arcsec=2.0):
    coord = SkyCoord.from_name(target_name)
    
    job = Gaia.cone_search_async(
        coord,
        radius=radius_arcsec * u.arcsec
    )
    results = job.get_results()
    print ("results: ", results)
    return results


def query_gaia_by_radec(ra_deg, dec_deg, radius_arcsec=2.0):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    
    job = Gaia.cone_search_async(
        coord,
        radius=radius_arcsec * u.arcsec
    )
    results = job.get_results()
    return results


def select_best_gaia_match(gaia_table):
    if len(gaia_table) == 0:
        return None
    return gaia_table[np.argmin(gaia_table['phot_g_mean_mag'])]


gaia_tbl = query_gaia_by_name("KELT-9")
best = select_best_gaia_match(gaia_tbl)

print("Gaia Source ID:", best['source_id'])
print("RA, DEC:", best['ra'], best['dec'])
print("G mag:", best['phot_g_mean_mag'])

for column in gaia_tbl.columns:
    print(column) #display the columns in the table