
import numpy as np
from configs.channel_config import SpectroscopyChannel
import logging

def generate_dark_image(channel: SpectroscopyChannel):
    nx = channel.x_pixels
    ny = channel.y_pixels
    dark_noise = channel.dark_noise
    dark_current_sigma = channel.dark_current_sigma
    exptime_s = channel.exposure_s

    dark_base = np.random.normal(dark_noise, dark_current_sigma, size=(ny, nx)).astype(np.float32)
    dark = (dark_base + (dark_noise * exptime_s))

    logging.info("Dark image generated: channel=%s shape=(%d,%d) exposure_s=%g mean=%g std=%g min=%g max=%", channel.channel_name, ny, nx, exptime_s, dark.mean(), dark.std(), dark.min(), dark.max())
    return dark