
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

def query_gaia_by_name(target_name, radius_arcsec=2.0):
    coord = SkyCoord.from_name(target_name)
    
    job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
    results = job.get_results()
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

def getcol(row, name):
    # try exact, then lowercase, then uppercase
    for k in (name, name.lower(), name.upper()):
        if k in row.colnames:
            return row[k]
    raise KeyError(f"Column {name!r} not found. Available: {row.colnames}")


gaia_tbl = query_gaia_by_name("KELT-9")
# print("gaia_tbl: ", gaia_tbl.colnames)

best = select_best_gaia_match(gaia_tbl)
print("Gaia Source ID:", getcol(best, "source_id"))
print("RA, DEC:", best['ra'], best['dec'])
print("G mag:", best['phot_g_mean_mag'])


from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def gaia_all_properties_by_name(target_name: str, radius_arcsec: float = 2.0):
    # Resolve name -> sky position
    coord = SkyCoord.from_name(target_name)

    # Cone search in gaia_source to get candidate Gaia sources around that position
    job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
    gaia_tbl = job.get_results()
    if len(gaia_tbl) == 0:
        raise RuntimeError(f"No Gaia sources radius_arcsec: float = 2.0found within {radius_arcsec} arcsec for '{target_name}'")

    # Pick a "best" match: brightest in G among candidates
    best = gaia_tbl[np.argmin(gaia_tbl["phot_g_mean_mag"])]
    source_id = int(best["source_id"])
    print("source id: ", source_id)

    # Fetch everything Gaia has in gaia_source + astrophysical_parameters for that source_id
    query = f"""
    SELECT *
FROM gaiadr3.gaia_source AS gs

LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id

LEFT JOIN gaiadr3.astrophysical_parameters_supp AS apsup
    ON gs.source_id = apsup.source_id

LEFT JOIN gaiadr3.gaia_source_simulation AS sourcesim
    ON gs.source_id = sourcesim.source_id

LEFT JOIN gaiadr3.gaia_universe_model AS um
    ON gs.source_id = um.source_id

    LEFT JOIN gaiadr3.sso_observation AS sso
    ON gs.source_id = sso.source_id

    LEFT JOIN gaiadr3.xp_summary AS xpsum
    ON gs.source_id = xpsum.source_id


WHERE gs.source_id = {source_id}

    """
    job2 = Gaia.launch_job_async(query)
    full = job2.get_results()
    if len(full) == 0:
        raise RuntimeError(f"No joined row returned for source_id={source_id}")

    return {
        "target_name": target_name,
        "coord_icrs": coord,
        "cone_candidates": gaia_tbl,
        "best_match": best,
        "source_id": source_id,
        "full_row": full[0],
        "full_columns": full.colnames,
    }


if __name__ == "__main__":
    TARGET = "WASP-69" 

    out = gaia_all_properties_by_name(TARGET, radius_arcsec=2.0)

    print("Target:", out["target_name"])
    print("Resolved coord (ICRS):", out["coord_icrs"].to_string("hmsdms"))

    print("Chosen source_id:", out["source_id"])
    print("Best-match RA,Dec (deg):", float(out["best_match"]["ra"]), float(out["best_match"]["dec"]))
    print("Best-match G mag:", float(out["best_match"]["phot_g_mean_mag"]))

    row = out["full_row"]
    print("\n--- Gaia DR3 full row (gaia_source + astrophysical_parameters) ---")
    print("Number of columns:", len(out["full_columns"]))

    for col in out["full_columns"]:
        print(f"{col} = {row[col]}")





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