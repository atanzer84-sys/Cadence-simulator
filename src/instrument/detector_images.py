import numpy as np
import logging
from configs.global_config import get_global_config
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.time import Time
from utils.images import write_frames_fits, write_frames_png
from domain.star import Star
import loaders.run_setup


def generate_detector_images_and_write_fits(counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, nuv_cfg, vis_cfg, user_cfg, output_dir, star):

    global_cfg = get_global_config()
    n_bias_and_darkframes = global_cfg.n_bias_and_darkframes
    n_science_frames = global_cfg.n_science_frames_per_channel

    header = initialize_fits_header(star)

    if n_bias_and_darkframes > 0:
        logging.info("Creating Bias and Dark Frames.")
        print("Creating Bias and Dark Frames.")

        bias_nuv_frames, bias_nuv_headers = generate_bias_frames(nuv_cfg, n_bias_and_darkframes, header)
        bias_vis_frames, bias_vis_headers = generate_bias_frames(vis_cfg, n_bias_and_darkframes, header)
        dark_nuv_frames, dark_nuv_headers = generate_dark_frames(nuv_cfg, n_bias_and_darkframes, user_cfg.exposure_NUV_s, header)
        dark_vis_frames, dark_vis_headers = generate_dark_frames(vis_cfg, n_bias_and_darkframes, user_cfg.exposure_VIS_s, header)

        #write FITS
        write_frames_fits(bias_nuv_frames, bias_nuv_headers, "bias", nuv_cfg.channel_name, output_dir)
        write_frames_fits(bias_vis_frames, bias_vis_headers, "bias", vis_cfg.channel_name, output_dir)
        write_frames_fits(dark_nuv_frames, dark_nuv_headers, "dark", nuv_cfg.channel_name, output_dir)
        write_frames_fits(dark_vis_frames, dark_vis_headers, "dark", vis_cfg.channel_name, output_dir)

        if global_cfg.write_dark_and_bias_png:
            write_frames_png(bias_nuv_frames, bias_nuv_headers, "bias", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(bias_vis_frames, bias_vis_headers, "bias", vis_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(dark_nuv_frames, dark_nuv_headers, "dark", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(dark_vis_frames, dark_vis_headers, "dark", vis_cfg.channel_name, output_dir, show_stats=True)
    else:
        logging.info("BIAS and DARK: n_bias_and_darkframes=%d → skipped.", n_bias_and_darkframes)

    if n_science_frames > 0:
        science_nuv_frames, science_nuv_headers = build_science_frames(counts_s_pixel_convolved_nuv, nuv_cfg, n_science_frames, user_cfg.exposure_NUV_s, header)
        science_vis_frames, science_vis_headers = build_science_frames(counts_s_pixel_convolved_vis, vis_cfg, n_science_frames, user_cfg.exposure_VIS_s, header)
        # write science FITS
        write_frames_fits(science_nuv_frames, science_nuv_headers, "science", nuv_cfg.channel_name, output_dir)
        write_frames_fits(science_vis_frames, science_vis_headers, "science", vis_cfg.channel_name, output_dir)
        # Write PNGs
        if global_cfg.write_science_frames_png:
            write_frames_png(science_nuv_frames, science_nuv_headers, "science", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(science_vis_frames, science_vis_headers, "science", vis_cfg.channel_name, output_dir, show_stats=True)


    
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

def generate_bias_frames(channel_cfg, n_frames, base_header):

    logging.info("BIAS: generating %d bias frames for %s (%d x %d).", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels)

    bias_frames = []
    bias_headers = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "BIAS", "Type of observation"))
        header.append(("CHANNEL", channel_cfg.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Bias {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Bias {i+1}", "Observation ID"))

        frame, header = generate_bias_frame(channel_cfg, header)

        bias_frames.append(frame)
        bias_headers.append(header)

    return bias_frames, bias_headers

def generate_bias_frame(channel_cfg, header=None):
    '''
    Bias = Offset (bias_offset) + Gaussian Noise (read_noise)
    '''
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels
    bias_offset = channel_cfg.bias_offset
    read_noise = channel_cfg.read_noise

    bias = bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))

    logging.info("BIAS STATS %s mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, bias.mean(), bias.std(), bias.min(), bias.max())

    if header is not None:
        header.append(("MEAN",      float(bias.mean()),     "Mean value of the frame"))
        header.append(("MEDIAN",    float(np.median(bias)), "Median value of the frame"))
        header.append(("STDDEV",    float(bias.std()),      "Standard deviation of the frame"))
        header.append(("MAX",       float(bias.max()),      "Maximum value of the frame"))
        header.append(("MIN",       float(bias.min()),      "Minimum value of the frame"))
        header.append(("B_OFFSET",  float(bias_offset),     "Threshold bias value applied"))
        header.append(("RNOISE",    float(read_noise),      "Readout noise"))
        header.append(("EXPTIME",   0.0,                    "Exposure time of observation"))

    return bias, header

def generate_dark_frames(channel_cfg, n_frames, exptime_s, base_header):

    logging.info("DARK: generating %d dark frames for %s (%d x %d), exptime_s=%g.", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels, exptime_s)

    dark_frames = []
    dark_headers = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "DARK", "Type of observation"))
        header.append(("CHANNEL", channel_cfg.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Dark {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Dark {i+1}", "Observation ID"))

        frame, header = generate_dark_frame(channel_cfg, exptime_s, header)

        dark_frames.append(frame)
        dark_headers.append(header)

        logging.info("DARK STATS %s frame=%d mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, i, frame.mean(), frame.std(), frame.min(), frame.max())

    return dark_frames, dark_headers

def generate_dark_frame(channel_cfg, exptime_s, header=None):
    '''
    Dark = Bias + (dark_current * exptime)
    '''
    dark_noise = channel_cfg.dark_noise
    dark_current_sigma = channel_cfg.dark_current_sigma
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels

    bias, _ = generate_bias_frame(channel_cfg, header=None)
    
    dark_base = np.random.normal(dark_noise, dark_current_sigma, size=(ny, nx))

    dark = bias + dark_base + (dark_noise * exptime_s)

    logging.info("DARK STATS %s mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, dark.mean(), dark.std(), dark.min(), dark.max())

    if header is not None:
        header.append(("MEAN",     float(dark.mean()),      "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(dark)),  "Median value of the frame"))
        header.append(("STDDEV",   float(dark.std()),      "Standard deviation of the frame"))
        header.append(("MAX",      float(dark.max()),       "Maximum value of the frame"))
        header.append(("MIN",      float(dark.min()),       "Minimum value of the frame"))
        header.append(("DARKVAL",  float(dark_noise),     "Input dark value"))
        header.append(("DARKSIG",  float(dark_current_sigma), "Dark noise sigma (e-/s/pixel)"))
        header.append(("EXPTIME",  float(exptime_s),        "Exposure time of observation"))
        header.append(("B_OFFSET", float(channel_cfg.bias_offset), "Bias offset used to generate frame"))
        header.append(("RNOISE",   float(channel_cfg.read_noise),  "Read noise sigma used to generate frame"))
    return dark, header

def build_science_frames(counts_s_pixel_convolved, channel_cfg, n_frames, exposure_time_s, base_header):

    science_frames = []
    science_headers = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "SCIENCE", "Type of observation"))
        header.append(("CHANNEL", channel_cfg.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Science {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Science {i+1}", "Observation ID"))
        header.append(("EXPTIME",  float(exposure_time_s),        "Exposure time of observation"))

        # generate new detector image
        detector_image, header = spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel_cfg, header)

        # generate new bias and dark
        bias_frame, _ = generate_bias_frame(channel_cfg, header = None)
        dark_frame, _ = generate_dark_frame(channel_cfg, exposure_time_s, header = None)

        # combine into science
        science = detector_image * exposure_time_s + dark_frame + bias_frame
        science_frames.append(science)
        science_headers.append(header)

    return science_frames, science_headers

def spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel_cfg, header=None):

    counts = counts_s_pixel_convolved
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels

    if len(counts) != nx:
        msg = f"counts length {len(counts)} != nx {nx}"
        logging.error(msg)
        print(msg)
        raise ValueError(msg)

    # choose a fixed trace center (placeholder)
    y0 = ny // 2

    # choose a spatial width in pixels (placeholder)
    spatial_sigma_pix = 5.0

    # build a normalized vertical profile w[y] with sum(w)=1
    w = np.zeros(ny, dtype=np.float64)
    for y in range(ny):
        dy = y - y0
        w[y] = np.exp(-0.5 * (dy / spatial_sigma_pix) * (dy / spatial_sigma_pix))

    w_sum = w.sum()
    if w_sum <= 0.0:
        logging.info("vertical profile w_sum=%g, normalizing", w_sum)
        raise ValueError("vertical profile sum <= 0")
    for y in range(ny):
        w[y] = w[y] / w_sum

    # fill the 2D image: for each x-column, distribute counts[x] over y using w[y]
    image = np.zeros((ny, nx), dtype=np.float64)
    for x in range(nx):
        for y in range(ny):
            image[y, x] = counts[x] * w[y]

    if header is not None:
        header.append(("MEAN",     float(image.mean()),      "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(image)),  "Median value of the frame"))
        header.append(("MAX",      float(image.max()),       "Maximum value of the frame"))
        header.append(("MIN",      float(image.min()),       "Minimum value of the frame"))
        header.append(("DARKVAL",  float(channel_cfg.dark_noise),     "Input dark value"))
        header.append(("B_OFFSET", float(channel_cfg.bias_offset), "Bias offset used to generate frame"))
        header.append(("RNOISE",   float(channel_cfg.read_noise),  "Read noise sigma used to generate frame"))

    return image, header
