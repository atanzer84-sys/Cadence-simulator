
import numpy as np
from configs.channel_config import SpectroscopyChannel
import logging

def generate_dark_image(channel: SpectroscopyChannel):
    nx = channel.x_pixels
    ny = channel.y_pixels
    dark_noise = channel.dark_noise
    dark_current_sigma = channel.dark_current_sigma
    exptime_s = channel.exposure_s

    dark_base = np.random.normal(dark_noise, dark_current_sigma, size=(ny, nx))
    dark = (dark_base + (dark_noise * exptime_s))

    logging.info("DARK STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, dark.mean(), dark.std(), dark.min(), dark.max())

    return dark