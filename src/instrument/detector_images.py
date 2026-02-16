import numpy as np
import logging
from configs.global_config import get_global_config
from utils.images import write_frames_png

def generate_bias_dark_frames(nuv_cfg, vis_cfg, user_cfg, output_dir):

    global_cfg = get_global_config()
    n_frames = global_cfg.n_bias_and_darkframes

    if n_frames <= 0:
        logging.info("BIAS and DARK: n_bias_frames=%d → no bias frames generated.", n_frames)
        return [], []

    logging.info("Creating Bias and Dark Frames.")
    print("Creating Bias and Dark Frames.")

    bias_nuv_frames = generate_bias_frames(nuv_cfg, n_frames)
    bias_vis_frames = generate_bias_frames(vis_cfg, n_frames)
    
    if global_cfg.write_dark_and_bias_png:
        write_frames_png(bias_nuv_frames, "bias", nuv_cfg.channel_name, output_dir)
        write_frames_png(bias_vis_frames, "bias", vis_cfg.channel_name, output_dir)


    return bias_nuv_frames, bias_vis_frames


def generate_bias_frames(channel_cfg, n_frames):

    logging.info("BIAS: generating %d bias frames for %s (%d x %d).", n_frames, channel_cfg.channel_name, channel_cfg.x_pixels, channel_cfg.y_pixels)

    frames = []
    for i in range(n_frames):
        frame = generate_bias_frame(channel_cfg)
        frames.append(frame)
        logging.info("BIAS STATS %s frame=%d mean=%g std=%g min=%g max=%g", channel_cfg.channel_name, i, frame.mean(), frame.std(), frame.min(), frame.max())

    return frames


def generate_bias_frame(channel_cfg):
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels
    bias_offset = channel_cfg.bias_offset_e
    read_noise = channel_cfg.read_noise_e_rms_per_pix

    bias = bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))

    return bias
