
import numpy as np
import logging
from frame.frame_class import Frame
from frame.bias import generate_bias_frame
from configs.channel_config import SpectroscopyChannel


def generate_dark_frames(channel: SpectroscopyChannel, n_frames, base_header):

    logging.info("DARK: generating %d dark frames for %s (%d x %d), exptime_s=%g.", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, channel.exposure_s)
    print(f"Creating DARK Frames for channel {channel.channel_name}.")
    

    dark_frames = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "DARK", "Type of observation"))
        header.append(("CHANNEL", channel.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Dark {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Dark {i+1}", "Observation ID"))

        dark = generate_dark_frame(channel, header)

        dark_frames.append(dark)

    return dark_frames

def generate_dark_frame(channel: SpectroscopyChannel, header=None):
    '''
    Dark = Bias + (dark_current * exptime)
    '''
    dark_noise = channel.dark_noise
    dark_current_sigma = channel.dark_current_sigma
    nx = channel.x_pixels
    ny = channel.y_pixels
    ccd_gain = channel.ccd_gain
    exptime_s = channel.exposure_s
    
    bias_frame = generate_bias_frame(channel, header=None)
    bias = bias_frame.data

    dark_base = np.random.normal(dark_noise, dark_current_sigma, size=(ny, nx))

    dark = bias + (dark_base + (dark_noise * exptime_s)) * ccd_gain

    logging.info("DARK STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, dark.mean(), dark.std(), dark.min(), dark.max())

    if header is not None:
        header.append(("MEAN",     float(dark.mean()),              "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(dark)),          "Median value of the frame"))
        header.append(("STDDEV",   float(dark.std()),               "Standard deviation of the frame"))
        header.append(("MAX",      float(dark.max()),               "Maximum value of the frame"))
        header.append(("MIN",      float(dark.min()),               "Minimum value of the frame"))
        header.append(("DARKVAL",  float(dark_noise),               "Input dark value"))
        header.append(("DARKSIG",  float(dark_current_sigma),       "Dark noise sigma (e-/s/pixel)"))
        header.append(("EXPTIME",  float(exptime_s),                "Exposure time of observation"))
        header.append(("B_OFFSET", float(channel.bias_offset),      "Bias offset used to generate frame"))
        header.append(("RNOISE",   float(channel.read_noise),       "Bias Read noise used to generate frame"))
        header.append(("YCUT1",     0,                              "Bottom of science box extraction"))
        header.append(("YCUT2",     ny-1,                           "Top of science box extraction"))
        header.append(("CCDGAIN",  ccd_gain,                        "CCD gain"))

    return Frame(data=dark, header=header, frame_type="dark", channel_tag=channel.channel_name)