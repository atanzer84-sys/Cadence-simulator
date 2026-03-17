from datetime import datetime

from domain.star import Star
from flux.flux_calc import calculate_glon_glat
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits
import numpy as np
from configs.channel_config import SpectroscopyChannel

def initialize_fits_header(star: Star, timestamp: datetime):
    """
    Create a base FITS header with all fixed keys set.
    Uses the given timestamp (typically ctx.timestamp from RunContext) for EXP_STRT.
    """
    t = timestamp
    time = Time(t, scale='utc')
    ra_hex  = Angle(star.right_ascension, unit="deg").to_string(unit="hour", sep=":", precision=6, pad=True)
    dec_hex = Angle(star.declination, unit="deg").to_string(unit="deg",  sep=":", precision=6, alwayssign=True, pad=True)
    glon, glat = calculate_glon_glat(star.right_ascension, star.declination)
    
    header = fits.Header()
    header.append(("TELESCOP",  "WALTzER",                          "Telescope name"))
    header.append(("ROOTNAME",  "WALTzER_output",                   "Root directory"))
    header.append(("EXP_STRT",  t.isoformat(timespec='seconds'),    "Exposure start time in UT"))
    header.append(("PRGRM_ID",  "WALTzER",                          "Program ID"))
    header.append(("DATEOBS",   time.strftime("%Y-%m-%d"),          "Date of Observation"))
    header.append(("TIMEOBS",   time.strftime("%H:%M:%S"),          "Time of start of observation in UTC"))
    header.append(("JD",        time.jd,                            "Time in Julian date"))
    header.append(("MJD",       time.mjd,                           "Time in Modified Julian Date"))
    header.append(("TRGET",     star.name ,                         "Target Name"))
    header.append(("TARGT_ID",  star.name ,                         "Target ID"))
    header.append(("TARGT_D",   star.distance_pc ,                  "Target distance in pc"))
    header.append(("TARGT_MS",  star.effective_temperature ,        "Target effective temperature"))
    header.append(("VMAG",      star.v_magnitude ,                  "V magnitude of the target"))
    header.append(("RA",        float(round(star.right_ascension, 4)), "Right ascension (deg)"))
    header.append(("DEC",       float(round(star.declination, 4)),     "Declination (deg)"))
    header.append(("GLAT",      float(round(glat, 4)),                 "Galactic latitude (deg)"))
    header.append(("GLON",      float(round(glon, 4)),                 "Galactic longitude (deg)"))
    header.append(("RA_HEX",    ra_hex,                                "RA in hh:mm:ss.sss"))
    header.append(("GEO_LAT",   0,                                  "Geocentric latitude"))
    header.append(("GEO_LON",   0,                                  "Geocentric longitude"))
    header.append(("DEC_HEX",   dec_hex,                            "Declination in dd:mm:ss.sss"))
    header.append(("CCDTEMP",   -50.00,                             "CCD temperature"))


    return header


def append_image_stats_header(header, image) -> None:
    """
    Append basic statistics for a 2D image to an existing FITS header.
    """
    if header is None:
        return

    # Use limited precision so FITS cards stay within 80 characters.
    header.append(("MEAN",   float(round(image.mean(), 2)),      "Mean value of the frame"))
    header.append(("MEDIAN", float(round(np.median(image), 2)),  "Median value of the frame"))
    header.append(("STDDEV", float(round(image.std(), 2)),       "Standard deviation of the frame"))
    header.append(("MAX",    float(round(image.max(), 2)),       "Maximum value of the frame"))
    header.append(("MIN",    float(round(image.min(), 2)),       "Minimum value of the frame"))


def append_channel_frame_header(header, channel: SpectroscopyChannel, exptime_s: float, include_bias: bool = True, include_dark: bool = True) -> None:
    """
    Append repeated simulator keywords that depend on channel + exposure.
    Controlled by flags so bias/dark/science can share one function.
    """
    if header is None:
        return

    header.append(("EXPTIME", float(exptime_s),             "Exposure time (seconds) of observation"))
    header.append(("YCUT1",   int(0),                       "Bottom of science box extraction"))
    header.append(("YCUT2",   int(channel.y_pixels - 1),    "Top of science box extraction"))
    header.append(("CCDGAIN", float(channel.ccd_gain),      "CCD gain"))

    if include_bias:
        header.append(("B_OFFSET", float(channel.bias_offset), "Bias offset (e-) used to generate frame"))
        header.append(("RNOISE",   float(channel.read_noise),  "Bias Read noise used (e-) to generate frame"))

    if include_dark:
        header.append(("DARKSIG", float(channel.dark_current_sigma), "Dark noise sigma (e-/s) used to generate frame"))
        header.append(("DARKVAL", float(channel.dark_noise),         "Input dark value (e-/s) used to generate frame"))


def append_base_frame_header(base_header, filetype: str, channel: SpectroscopyChannel, index0: int):
    """
    Create a per-frame header copy and append the repeated identity keys.

    index0 is 0-based (your loop counter i).
    label defaults to filetype (so "DARK" -> "Dark" for EXP_ID/OBS_ID).
    """
    if base_header is None:
        return None

    header = base_header.copy()

    k = index0 + 1

    header.append(("FILETYPE",  filetype,               "Type of observation"))
    header.append(("CHANNEL",   channel.channel_name,   "Detector channel"))
    header.append(("EXP_ID",    f"{filetype} {k}",      "Exposure ID"))
    header.append(("OBS_ID",    f"Obs {filetype} {k}",  "Observation ID"))

    return header

def append_photometry_header(header, phot) -> None:
    if header is None or phot is None:
        return

    counts_star, counts_star_noise, x0, y0, radius_annulus_inner, radius_annulus_outer = phot

    # Limit precision to keep FITS cards within 80 characters.
    header.append(("CSTAR", float(round(counts_star, 2)), "Aperture stellar counts (e-)"))
    header.append(("CSTNOISE", float(round(counts_star_noise, 2)), "Noise of CSTAR (e-)"))

    header.append(("PHOTX0", int(x0), "Photometry center X (pix)"))
    header.append(("PHOTY0", int(y0), "Photometry center Y (pix)"))

    header.append(("PHOTINR", float(round(radius_annulus_inner, 2)), "BKG annulus inner radius"))
    header.append(("PHOTOUTR", float(round(radius_annulus_outer, 2)), "BKG annulus outer radius"))