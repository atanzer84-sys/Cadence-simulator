from datetime import datetime

def ts(msg):
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], msg)

ts("start")
# from astroquery.gaia import Gaia
ts("nach astroquery")

# from astropy.coordinates import SkyCoord
ts("nach astropy")

ts("fertig")