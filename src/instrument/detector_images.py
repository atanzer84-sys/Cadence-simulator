import numpy as np

def generate_bias_frames(channel_cfg, n_frames, rng):
    """
    Generate n_frames bias frames (in electrons).

    bias_offset_e: constant offset (mean level)
    read_noise_e_rms_per_pix: Gaussian sigma per pixel
    """
    nx = int(channel_cfg.x_pixels)
    ny = int(channel_cfg.y_pixels)

    bias_offset_e = float(channel_cfg.bias_offset_e)
    sigma_read_e = float(channel_cfg.read_noise_e_rms_per_pix)

    frames = bias_offset_e + rng.normal(0.0, sigma_read_e, size=(n_frames, ny, nx))
    master = frames.mean(axis=0)

    return frames, master
