import numpy as np
import logging
from frame.dark import generate_dark_frame
from configs.channel_config import SpectroscopyChannel
from instrument.spectrum_spread import spread_1d_spectrum_to_2d
from frame.frame_class import Frame

def generate_science_frame(counts_s_pixel_convolved, channel: SpectroscopyChannel, header):
    exposure_time_s = channel.exposure_s
    ccd_gain = channel.ccd_gain

    detector_image, header = spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel, header)

    dark = generate_dark_frame(channel, header=None)
    if isinstance(dark, Frame):
        dark = dark.data

    science = dark + (detector_image * exposure_time_s) * ccd_gain

    header.append(("MEAN",   float(science.mean()),     "Mean value of the frame"))
    header.append(("MEDIAN", float(np.median(science)), "Median value of the frame"))
    header.append(("STDDEV", float(science.std()),      "Standard deviation of the frame"))
    header.append(("MAX",    float(science.max()),      "Maximum value of the frame"))
    header.append(("MIN",    float(science.min()),      "Minimum value of the frame"))

    return Frame(data=science, header=header, frame_type="science", channel_tag=channel.channel_name)


def generate_science_frames(counts_s_pixel_convolved, channel: SpectroscopyChannel, n_frames, base_header):
    exposure_time_s = channel.exposure_s
    logging.info("SCIENCE: generating %d science frames for %s (%d x %d), exptime_s=%g.", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, exposure_time_s)
    print(f"Creating SCIENCE Frames for channel {channel.channel_name}.")

    frames = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "SCIENCE", "Type of observation"))
        header.append(("CHANNEL", channel.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Science {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Science {i+1}", "Observation ID"))
        header.append(("EXPTIME", float(exposure_time_s), "Exposure time of observation"))
        header.append(("YCUT1", 0, "Bottom of science box extraction"))
        header.append(("YCUT2", channel.y_pixels - 1, "Top of science box extraction"))
        header.append(("CCDGAIN", channel.ccd_gain, "CCD gain"))
        header.append(("B_OFFSET", float(channel.bias_offset), "Bias offset used to generate frame"))
        header.append(("RNOISE", float(channel.read_noise), "Bias Read noise used to generate frame"))
        header.append(("DARKSIG", float(channel.dark_current_sigma), "Dark noise sigma used to generate frame"))
        header.append(("DARKVAL", float(channel.dark_noise), "Input dark value used to generate frame"))

        frame = generate_science_frame(counts_s_pixel_convolved, channel, header)
        frames.append(frame)

    return frames