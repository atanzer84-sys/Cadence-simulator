
import numpy as np
import logging

def generate_bias_frames(channel_cfg, n_frames, base_header):

    logging.info("BIAS: generating %d bias frames for %s (%d x %d).", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels)
    print(f"Creating BIAS Frames for channel {channel_cfg.channel_name}.")

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
    ccd_gain = channel_cfg.ccd_gain

    bias = (bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))) * ccd_gain

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
        header.append(("YCUT1",     0,                      "Bottom of science box extraction"))
        header.append(("YCUT2",     ny-1,                   "Top of science box extraction"))
        header.append(("CCDGAIN",   ccd_gain,               "CCD gain"))

    return bias, header