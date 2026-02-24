

import numpy as np
import logging
from frame.frame_class import Frame
from configs.channel_config import SpectroscopyChannel
from frame.spectra import generate_spectra_frame
from instrument.photon_noise import apply_photon_noise_gauss_from_spectra2d

def generate_photon_noise_frames(frame: Frame, channel: SpectroscopyChannel, n_frames, base_header):

    logging.info("PHOTON NOISE: generating %d noise frames for %s (%d x %d).", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels)
    print(f"Creating PHOTON NOISE Frames for channel {channel.channel_name}.")

    photon_noise_frames = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "PHOTON NOISE", "Type of observation"))
        header.append(("CHANNEL", channel.channel_name, "Detector channel"))
        header.append(("EXP_ID", f"PHOTON NOISE {i+1}", "Exposure ID"))
        header.append(("OBS_ID", f"Obs PHOTON NOISE {i+1}", "Observation ID"))

        photon_noise = generate_photon_noise_frame(frame, channel, header)

        photon_noise_frames.append(photon_noise)

    return photon_noise_frames

def generate_photon_noise_frame(frame: Frame, channel: SpectroscopyChannel, header=None):

    spectra_data_with_exposure_and_gain = frame.data

    photon_noise = apply_photon_noise_gauss_from_spectra2d(spectra_data_with_exposure_and_gain, channel)

    if header is not None:
        header.append(("MEAN", float(photon_noise.mean()),          "Mean value of the frame"))
        header.append(("MEDIAN", float(np.median(photon_noise)),    "Median value of the frame"))
        header.append(("STDDEV", float(photon_noise.std()),         "Standard deviation of the frame"))
        header.append(("MAX", float(photon_noise.max()),            "Maximum value of the frame"))
        header.append(("MIN", float(photon_noise.min()),            "Minimum value of the frame"))
        header.append(("EXPTIME", channel.exposure,                 "Exposure time of observation"))
        header.append(("YCUT1", 0,                                  "Bottom of science box extraction"))
        header.append(("YCUT2", channel.y_pixels - 1,               "Top of science box extraction"))
        header.append(("CCDGAIN", channel.ccd_gain,                 "CCD gain"))
        header.append(("B_OFFSET", float(channel.bias_offset),      "Bias offset used to generate frame"))
        header.append(("RNOISE", float(channel.read_noise),         "Bias Read noise used to generate frame"))
        header.append(("DARKSIG", float(channel.dark_current_sigma),"Dark noise sigma used to generate frame"))
        header.append(("DARKVAL", float(channel.dark_noise),        "Input dark value used to generate frame"))

    return Frame(data=photon_noise, header=header, frame_type="photon_noise", channel_tag=channel.channel_name)