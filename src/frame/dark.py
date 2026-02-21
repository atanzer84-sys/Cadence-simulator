
import numpy as np
import logging
from frame.bias import generate_bias_frame


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
    ccd_gain = channel_cfg.ccd_gain

    bias, _ = generate_bias_frame(channel_cfg, header=None)
    
    dark_base = np.random.normal(dark_noise, dark_current_sigma, size=(ny, nx))

    dark = (bias + dark_base + (dark_noise * exptime_s)) * ccd_gain

    logging.info("DARK STATS %s mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, dark.mean(), dark.std(), dark.min(), dark.max())

    if header is not None:
        header.append(("MEAN",     float(dark.mean()),              "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(dark)),          "Median value of the frame"))
        header.append(("STDDEV",   float(dark.std()),               "Standard deviation of the frame"))
        header.append(("MAX",      float(dark.max()),               "Maximum value of the frame"))
        header.append(("MIN",      float(dark.min()),               "Minimum value of the frame"))
        header.append(("DARKVAL",  float(dark_noise),               "Input dark value"))
        header.append(("DARKSIG",  float(dark_current_sigma),       "Dark noise sigma (e-/s/pixel)"))
        header.append(("EXPTIME",  float(exptime_s),                "Exposure time of observation"))
        header.append(("B_OFFSET", float(channel_cfg.bias_offset),  "Bias offset used to generate frame"))
        header.append(("RNOISE",   float(channel_cfg.read_noise),   "Read noise sigma used to generate frame"))
        header.append(("CCDGAIN",  ccd_gain,                        "CCD gain"))

    return dark, header