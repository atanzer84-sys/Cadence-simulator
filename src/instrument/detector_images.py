import numpy as np
import logging
from configs.global_config import get_global_config
from utils.images import write_frames_png

def generate_bias_dark_frames(nuv_cfg, vis_cfg, user_cfg, output_dir):

    global_cfg = get_global_config()
    n_frames = global_cfg.n_bias_and_darkframes

    if n_frames <= 0:
        logging.info("BIAS and DARK: n_bias_frames=%d → no bias frames generated.", n_frames)
        return [], [], [], []

    logging.info("Creating Bias and Dark Frames.")
    print("Creating Bias and Dark Frames.")

    bias_nuv_frames = generate_bias_frames(nuv_cfg, n_frames)
    bias_vis_frames = generate_bias_frames(vis_cfg, n_frames)
    dark_nuv_frames = generate_dark_frames(nuv_cfg, n_frames, user_cfg.exposure_NUV_s)
    dark_vis_frames = generate_dark_frames(vis_cfg, n_frames, user_cfg.exposure_VIS_s)

    
    if global_cfg.write_dark_and_bias_png:
        write_frames_png(bias_nuv_frames, "bias", nuv_cfg.channel_name, output_dir)
        write_frames_png(bias_vis_frames, "bias", vis_cfg.channel_name, output_dir)
        write_frames_png(dark_nuv_frames, "dark", nuv_cfg.channel_name, output_dir)
        write_frames_png(dark_vis_frames, "dark", vis_cfg.channel_name, output_dir)        


    return bias_nuv_frames, bias_vis_frames, dark_nuv_frames, dark_vis_frames


def generate_bias_frames(channel_cfg, n_frames):

    logging.info("BIAS: generating %d bias frames for %s (%d x %d).", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels)

    bias_frames = []
    for i in range(n_frames):
        frame = generate_bias_frame(channel_cfg)
        bias_frames.append(frame)
        logging.info("BIAS STATS %s frame=%d mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, i, frame.mean(), frame.std(), frame.min(), frame.max())

    return bias_frames

def generate_bias_frame(channel_cfg):
    '''
    Bias = Offset (bias_offset) + Gaussian Noise (read_noise)
    '''
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels
    bias_offset = channel_cfg.bias_offset
    read_noise = channel_cfg.read_noise

    bias = bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))

    return bias

def generate_dark_frames(channel_cfg, n_frames, exptime_s):

    logging.info("DARK: generating %d dark frames for %s (%d x %d), exptime_s=%g.", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels, exptime_s)

    dark_frames = []

    for i in range(n_frames):
        frame = generate_dark_frame(channel_cfg, exptime_s)
        dark_frames.append(frame)
        logging.info("DARK STATS %s frame=%d mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, i, frame.mean(), frame.std(), frame.min(), frame.max())

    return dark_frames

def generate_dark_frame(channel_cfg, exptime_s):
    '''
    Dark = Bias + (dark_current * exptime)
    '''

    dark_current = channel_cfg.dark_noise

    bias = generate_bias_frame(channel_cfg)
    dark_signal = dark_current * exptime_s

    dark = bias + dark_signal

    return dark
