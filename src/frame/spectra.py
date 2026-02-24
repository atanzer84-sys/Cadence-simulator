
import numpy as np
import logging
from frame.frame_class import Frame
from configs.channel_config import SpectroscopyChannel
from frame.dark_frame import generate_dark_frame

def generate_specta_frames(spectra_2d, channel: SpectroscopyChannel, n_frames, base_header):

    logging.info("SPECTRA: generating %d spectra frames for %s (%d x %d).", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels)
    print(f"Creating Spectra Frames for channel {channel.channel_name}.")

    spectra_frames = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "SPECTRA", "Type of observation"))
        header.append(("CHANNEL", channel.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"Spectra {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs Spectra {i+1}", "Observation ID"))

        spectra = generate_spectra_frame(spectra_2d, channel, header)

        spectra_frames.append(spectra)

    return spectra_frames

def generate_spectra_frame(spectra_2d, channel: SpectroscopyChannel, header=None):

    ccd_gain = channel.ccd_gain
    exposure = channel.exposure_s
    dark_frame = generate_dark_frame(channel, header=None)
    dark = dark_frame.data

    spectra = dark + (spectra_2d * exposure * ccd_gain)

    if header is not None:
        header.append(("MEAN", float(spectra.mean()), "Mean value of the frame"))
        header.append(("MEDIAN", float(np.median(spectra)), "Median value of the frame"))
        header.append(("STDDEV", float(spectra.std()), "Standard deviation of the frame"))
        header.append(("MAX", float(spectra.max()), "Maximum value of the frame"))
        header.append(("MIN", float(spectra.min()), "Minimum value of the frame"))
        header.append(("EXPTIME", exposure, "Exposure time of observation"))
        header.append(("YCUT1", 0, "Bottom of science box extraction"))
        header.append(("YCUT2", channel.y_pixels - 1, "Top of science box extraction"))
        header.append(("CCDGAIN", ccd_gain, "CCD gain"))
        header.append(("B_OFFSET", float(channel.bias_offset), "Bias offset used to generate frame"))
        header.append(("RNOISE", float(channel.read_noise), "Bias Read noise used to generate frame"))
        header.append(("DARKSIG", float(channel.dark_current_sigma), "Dark noise sigma used to generate frame"))
        header.append(("DARKVAL", float(channel.dark_noise), "Input dark value used to generate frame"))

    return Frame(data=spectra, header=header, frame_type="spectra", channel_tag=channel.channel_name)