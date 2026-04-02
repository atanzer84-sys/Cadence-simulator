
import numpy as np
from configs.channel_config import Channel

# TODO: FLAT IMPLEMENTATION HERE
def generate_flat_image(channel: Channel):
    nx = channel.x_pixels
    ny = channel.y_pixels
    flat = np.ones((ny, nx), dtype=np.float32)

    
    # logging.info("FLAT IMAGE %s (%d x %d): unity placeholder -> mean=%g std=%g min=%g max=%g", channel.channel_name, nx, ny, flat.mean(), flat.std(), flat.min(), flat.max())
    
    return flat 