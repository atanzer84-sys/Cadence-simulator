import numpy as np
import logging
from configs.global_config import get_global_config
from utils.images import write_frames_png

def generate_bias_dark_frames(nuv_cfg, vis_cfg, user_cfg, output_dir):

    global_cfg = get_global_config()
    n_bias_frames = global_cfg.n_bias_frames

    if n_bias_frames <= 0:
        logging.info("BIAS: n_bias_frames = %d → no bias frames will be generated.", n_bias_frames)
        return [], []

    logging.info("Creating Bias and Dark Frames.")
    print("Creating Bias and Dark Frames.")

    bias_nuv_frames = generate_bias_frames(nuv_cfg, n_bias_frames)
    bias_vis_frames = generate_bias_frames(vis_cfg, n_bias_frames)
    
    if global_cfg.write_dark_and_bias_png:
        write_frames_png(bias_nuv_frames, "bias", nuv_cfg.channel_name, output_dir)
        write_frames_png(bias_vis_frames, "bias", vis_cfg.channel_name, output_dir)


    return bias_nuv_frames, bias_vis_frames


def generate_bias_frames(channel_cfg, n_bias_frames):

    logging.info("BIAS: generating %d bias frames (%d x %d).", n_bias_frames, channel_cfg.x_pixels,channel_cfg.y_pixels)

    frames = []

    for _ in range(n_bias_frames):
        frame = generate_bias_frame(channel_cfg)
        frames.append(frame)

    return frames


def generate_bias_frame(channel_cfg):
    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels
    bias_offset = channel_cfg.bias_offset_e
    read_noise = channel_cfg.read_noise_e_rms_per_pix

    bias = bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))

    return bias
