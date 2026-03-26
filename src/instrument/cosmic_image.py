import logging
import numpy as np
from configs.channel_config import Channel
from configs.global_config import GlobalConfig

_rng = np.random.default_rng()

def generate_cosmic_rays(channel: Channel, cfg: GlobalConfig):

    nx = channel.x_pixels
    ny = channel.y_pixels
    min_rays = cfg.cosmic_rays_min
    max_rays = cfg.cosmic_rays_max
    cosmic_ray_charge_e = cfg.cosmic_ray_signal_electrons
    cosmic_ray_length_min_px = cfg.cosmic_ray_length_min_px
    cosmic_ray_length_max_px = cfg.cosmic_ray_length_max_px

    cosmic_rays = _rng.integers(min_rays, max_rays + 1)
    image = np.zeros((ny, nx), dtype=np.float32)

    # IDL: x_val=(nx-1)*randomu(seed,number) etc.
    x_val = (nx - 1) * _rng.random(cosmic_rays)
    y_val = (ny - 1) * _rng.random(cosmic_rays)

    direction = np.deg2rad(360.0 * _rng.random(cosmic_rays))

    for i in range(cosmic_rays):
        length_cosmic_ray = _rng.integers(cosmic_ray_length_min_px, cosmic_ray_length_max_px + 1)

        x = int(np.trunc(x_val[i]))
        y = int(np.trunc(y_val[i]))

        image[y, x] = cosmic_ray_charge_e

        for j in range(length_cosmic_ray):
            x_shft = int(np.round(j * np.cos(direction[i])))
            y_shft = int(np.round(j * np.sin(direction[i])))

            x_n = x + x_shft
            y_n = y + y_shft

            # skip if outside detector
            if not (0 <= x_n < nx and 0 <= y_n < ny):
                break

            image[y_n, x_n] = cosmic_ray_charge_e

    logging.info("Cosmic rays generated: channel=%s count=%d signal_e=%d", channel.channel_name, cosmic_rays, cosmic_ray_charge_e)
    return image


