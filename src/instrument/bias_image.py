
import numpy as np
from configs.channel_config import SpectroscopyChannel
import logging

def generate_bias_image(channel: SpectroscopyChannel):
    nx = channel.x_pixels
    ny = channel.y_pixels
    bias_offset = channel.bias_offset
    read_noise = channel.read_noise
    ccd_gain = channel.ccd_gain

    bias = (bias_offset + np.random.normal(0.0, read_noise, size=(ny, nx))).astype(np.float32)
    
    logging.info("BIAS IMAGE %s (%d x %d): bias_offset=%g read_noise=%g ccd_gain=%g -> mean=%g std=%g min=%g max=%g", channel.channel_name, nx, ny, bias_offset, read_noise, ccd_gain, bias.mean(), bias.std(), bias.min(), bias.max())
    return bias