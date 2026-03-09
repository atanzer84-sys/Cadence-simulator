import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel
from domain.star import Star
from domain.star_catalog import StarCatalog
from instrument.psf_spread import get_photometry_placement, paste_psf_stamp
import logging

def generate_background_star_photometry_image(channel: PhotometryChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog, roll_angle_start: float, roll_angle_stop: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_arcs: dict[str, list[tuple[int, int]]] = {}

    
    detector_start = _build_detector(channel, roll_angle_start)
    detector_end = _build_detector(channel, roll_angle_stop)
    target_star_placement = get_photometry_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_on_detector = 0

    for star_id in background_stars_catalog.stars_by_id:
        bg_star_result = _render_star_if_on_detector(star_id, channel, background_stars_catalog, frame_index, detector_start, detector_end, target_star_placement, roll_angle_start, roll_angle_stop)
        
        if bg_star_result is None:
            continue

        bg_star_image, valid_positions = bg_star_result
        image += bg_star_image
        background_star_arcs[star_id] = valid_positions
        n_on_detector += 1

    logging.info("BG STARS ARC: frame=%d channel=%s roll_angle_start=%g roll_angle_stop=%g n_on_detector=%d/%d image_sum=%g", frame_index, channel.channel_name, float(roll_angle_start), float(roll_angle_stop), int(n_on_detector), int(total), float(np.sum(image)))

    return image, background_star_arcs

def _build_detector(channel: PhotometryChannel, roll_angle_deg: float) -> tuple[float, float, float, float]:
    rad = np.deg2rad(float(roll_angle_deg))
    cos_roll = float(np.cos(rad))
    sin_roll = float(np.sin(rad))
    half_w = float(channel.x_pixels * channel.pixel_scale * 0.5)
    half_h = float(channel.y_pixels * channel.pixel_scale * 0.5)
    return cos_roll, sin_roll, half_w, half_h

def _render_star_if_on_detector(star_id: str, channel: PhotometryChannel, catalog: StarCatalog, frame_index: int, detector_start: tuple[float, float, float, float], detector_end: tuple[float, float, float, float], target_star_placement: tuple[int, float], roll_angle_start, roll_angle_stop) -> tuple[np.ndarray, list[tuple[int, int]]] | None:

    x_target, y_target = target_star_placement
    dx, dy = catalog.get_offset_arcsec(star_id)

    total_flux_electrons_per_second = _get_cached_counts(star_id, catalog, channel, frame_index)
    if total_flux_electrons_per_second is None:
        return None

    roll_angles = _compute_roll_angle_samples(dx, dy, channel, roll_angle_start, roll_angle_stop)
    star_image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    valid_positions: list[tuple[int, int]] = []

    for roll_angle_deg in roll_angles:
        detector = _build_detector(channel, roll_angle_deg)
        cos_roll, sin_roll, half_w, half_h = detector
        uv = _detector_check(dx, dy, cos_roll, sin_roll, half_w, half_h)
        if uv is None:
            continue
        u, v = uv
        x_background_star, y_background_star = _detector_position(x_target, y_target, u, v, channel)
        valid_positions.append((x_background_star, y_background_star))

    if len(valid_positions) == 0:
        return None

    # exposure will not change, star is always visible and therefore for the full exposure.
    flux_per_step_electrons_per_second = total_flux_electrons_per_second / float(len(valid_positions))

    psf_center_x = int(channel.psf_center_x)
    psf_center_y = int(channel.psf_center_y)

    for x_background_star, y_background_star in valid_positions:
        psf_stamp = channel.psf_image * flux_per_step_electrons_per_second
        paste_psf_stamp(star_image, psf_stamp, x_background_star, y_background_star, psf_center_x, psf_center_y)

    return star_image, valid_positions

def _detector_check(dx: float, dy: float, cos_roll: float, sin_roll: float, half_w: float, half_h: float) -> tuple[float, float] | None:
    u = dx * cos_roll + dy * sin_roll
    v = -dx * sin_roll + dy * cos_roll

    if abs(u) > half_w:
        return None
    if abs(v) > half_h:
        return None

    return u, v

def _get_cached_counts(star_id: str, catalog: StarCatalog, channel: PhotometryChannel, frame_index: int) -> np.ndarray | None:
    key = (star_id, channel.channel_name)
    if key not in catalog.counts_by_id_and_band:
        logging.info("BG STAR missing cached counts: frame=%d channel=%s star_id=%s", frame_index, channel.channel_name, star_id)
        return None
    return catalog.counts_by_id_and_band[key]

def _detector_position(x_target: int, y_target: int, u: float, v: float, channel: PhotometryChannel) -> tuple[int, int]:
    x_background_star = int(round(x_target + u / channel.pixel_scale))
    y_background_star = int(round(y_target + v / channel.pixel_scale))
    return x_background_star, y_background_star

def _compute_roll_angle_samples(dx: float, dy: float, channel: PhotometryChannel, roll_angle_start: float, roll_angle_stop: float, max_motion_per_step_px: float = 0.25) -> np.ndarray:
    radius_arcsec = float(np.hypot(dx, dy))
    delta_angle_rad = float(abs(np.deg2rad(roll_angle_stop - roll_angle_start)))
    arc_length_arcsec = radius_arcsec * delta_angle_rad
    arc_length_px = arc_length_arcsec / float(channel.pixel_scale)
    n_steps = max(2, int(np.ceil(arc_length_px / max_motion_per_step_px)) + 1)

    roll_angles = np.linspace(roll_angle_start, roll_angle_stop, n_steps)

    # logging.info("BG STAR ARC SAMPLING | dx=%f dy=%f radius_arcsec=%f delta_angle_deg=%f arc_length_arcsec=%f arc_length_px=%f n_steps=%d roll_angles=%s", dx, dy, radius_arcsec, roll_angle_stop - roll_angle_start, arc_length_arcsec, arc_length_px, n_steps, roll_angles)
    
    return roll_angles


