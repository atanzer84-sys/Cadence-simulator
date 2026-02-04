# from astroquery.gaia import Gaia
# from astropy.coordinates import SkyCoord
# import astropy.units as u

# TODO: GAIA LOOKUP
def lookup_star_gaia(star_params: dict) -> dict:
    try:

        return {} 
    except Exception as e:
        print("Gaia lookup failed:", e)
        return {}
