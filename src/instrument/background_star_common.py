import numpy as np
import logging
from configs.channel_config import Channel
from domain.star_catalog import StarCatalog


def compute_roll_angle_samples(dx: float, dy: float, channel: Channel, roll_angle_start: float, roll_angle_stop: float, max_motion_per_step_px: float = 0.25) -> np.ndarray:
    radius_arcsec = float(np.hypot(dx, dy))
    delta_angle_rad = float(abs(np.deg2rad(roll_angle_stop - roll_angle_start)))
    arc_length_arcsec = radius_arcsec * delta_angle_rad
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

def check_within_rotated_bounds(dx: float, dy: float, bounds: tuple[float, float, float, float]) -> tuple[float, float] | None:
    cos_roll, sin_roll, half_w, half_h = bounds

    u = dx * cos_roll + dy * sin_roll
    v = -dx * sin_roll + dy * cos_roll
    if abs(u) > half_w or abs(v) > half_h:
        return None
    return u, v


def build_rotated_bounds(half_bounds: tuple[float, float], roll_angle_deg: float) -> tuple[float, float, float, float]:
    half_w, half_h = half_bounds

    rad = np.deg2rad(float(roll_angle_deg))
    cos_roll = float(np.cos(rad))
    sin_roll = float(np.sin(rad))

    return cos_roll, sin_roll, half_w, half_h