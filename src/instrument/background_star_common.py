import numpy as np
import logging
from configs.channel_config import Channel
from domain.star_catalog import StarCatalog


def compute_roll_angle_samples(separation: float, channel: Channel, roll_angle_start: float, roll_angle_stop: float, max_motion_per_step_px: float = 0.75) -> np.ndarray:
    delta_angle_rad = float(abs(np.deg2rad(roll_angle_stop - roll_angle_start)))
    arc_length_arcsec = separation * delta_angle_rad
    arc_length_px = arc_length_arcsec / float(channel.pixel_scale)
    n_steps = max(2, int(np.ceil(arc_length_px / max_motion_per_step_px)) + 1)

    roll_angles = np.linspace(roll_angle_start, roll_angle_stop, n_steps, dtype=np.float32)

    return roll_angles


def get_cached_counts(star_id: str, catalog: StarCatalog, channel: Channel, frame_index: int) -> np.ndarray | None:
    key = (star_id, channel.channel_name)
    if key not in catalog.counts_by_id_and_band:
        logging.info("BG STAR missing cached counts: frame=%d channel=%s star_id=%s", frame_index, channel.channel_name, star_id)
        return None
    return catalog.counts_by_id_and_band[key]

def rotated_coordinates(dx: float, dy: float, roll_angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(roll_angles_deg.astype(np.float32))
    cos_r = np.cos(theta)
    sin_r = np.sin(theta)

    u = dx * cos_r + dy * sin_r
    v = -dx * sin_r + dy * cos_r

    return u, v

def within_rotated_bounds_mask(u: np.ndarray, v: np.ndarray, half_bounds: tuple[float, float]) -> np.ndarray:
    half_w, half_h = half_bounds
    return (np.abs(u) <= half_w) & (np.abs(v) <= half_h)