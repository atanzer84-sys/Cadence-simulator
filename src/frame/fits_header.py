from domain.star import Star
import loaders.run_setup
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits



def initialize_fits_header(star: Star):
    """
    Create a base FITS header with all fixed keys set.
    Uses the Output Directory timestamp for EXP_STRT.
    """
    t = loaders.run_setup.GLOBAL_TIMESTAMP
    time = Time(t, scale='utc')

    header = fits.Header()
    header.append(("TELESCOP", "WALTzER", "Telescope name"))
    header.append(("ROOTNAME", "WALTzER_output", "Root directory"))
    header.append(("EXP_STRT", t.isoformat(timespec='seconds'), "Exposure start time in UT"))
    header.append(("DATEOBS", time.strftime("%Y-%m-%d"), "Date of Observation"))
    header.append(("TIMEOBS", time.strftime("%H:%M:%S"), "Time of start of observation in UTC"))
    header.append(("JD", time.jd, "Time in Julian date"))
    header.append(("MJD", time.mjd, "Time in Modified Julian Date"))

    ra_hex  = Angle(star.right_ascension, unit="deg").to_string(unit="hour", sep=":", precision=6, pad=True)
    dec_hex = Angle(star.declination, unit="deg").to_string(unit="deg",  sep=":", precision=6, alwayssign=True, pad=True)
    header.append(("TARGT_ID", star.name , "Target Name"))
    header.append(("TARGT_D", star.distance_pc , "Target distance in pc"))
    header.append(("TARGT_MS", star.mass_sun_kg , "Target mass in Sun_mass in kg"))
    header.append(("RA", star.right_ascension, "Right ascension in degrees"))
    header.append(("DEC", star.declination, "Declination in degrees"))
    header.append(("RA_HEX", ra_hex, "Right ascension in hh:mm:ss.sss"))
    header.append(("DEC_HEX", dec_hex, "Declination in dd:mm:ss.sss"))
    header.append(("CCDTEMP", -50.00, "CCD temperature"))


    return header