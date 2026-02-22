import numpy as np
import logging
from frame.dark import generate_dark_frame
from configs.channel_config import SpectroscopyChannel
from detector.spectrum_spread import spread_1d_spectrum_to_2d

def generate_science_frames(counts_s_pixel_convolved, channel: SpectroscopyChannel, n_frames, base_header):

    exposure_time_s = channel.exposure_s
    logging.info("SCIENCE: generating %d science frames for %s (%d x %d), exptime_s=%g.", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, exposure_time_s)
    print(f"Creating SCIENCE Frames for channel {channel.channel_name}.")
    ccd_gain = channel.ccd_gain

    science_frames = []
    science_headers = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE",  "SCIENCE",                          "Type of observation"))
        header.append(("CHANNEL",   channel.channel_name,               "Detector channel"))
        header.append(("EXP_ID",    f"Science {i+1}",                   "Exposure ID"))
        header.append(("OBS_ID",    f"Obs Science {i+1}",               "Observation ID"))
        header.append(("EXPTIME",   float(exposure_time_s),             "Exposure time of observation"))
        header.append(("YCUT1",     0,                                  "Exposure time of observation"))
        header.append(("YCUT2",     channel.y_pixels-1,                 "Exposure time of observation"))
        header.append(("CCDGAIN",   ccd_gain,                           "CCD gain"))
        header.append(("B_OFFSET",  float(channel.bias_offset),         "Bias offset used to generate frame"))
        header.append(("RNOISE",    float(channel.read_noise),          "Bias Read noise used to generate frame"))
        header.append(("DARKSIG",   float(channel.dark_current_sigma),  "Dark noise sigma used to generate frame"))
        header.append(("DARKVAL",   float(channel.dark_noise),          "Input dark value used to generate frame"))
 
        # generate new detector image
        detector_image, header = spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel, header)

        # generate new bias and dark
        dark_frame, _ = generate_dark_frame(channel, header = None)

        # combine into science
        science = dark_frame + (detector_image * exposure_time_s) * ccd_gain
        header.append(("MEAN",     float(science.mean()),             "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(science)),         "Median value of the frame"))
        header.append(("STDDEV",   float(science.std()),              "Standard deviation of the frame"))
        header.append(("MAX",      float(science.max()),              "Maximum value of the frame"))
        header.append(("MIN",      float(science.min()),              "Minimum value of the frame"))
        science_frames.append(science)
        science_headers.append(header)

    return science_frames, science_headers