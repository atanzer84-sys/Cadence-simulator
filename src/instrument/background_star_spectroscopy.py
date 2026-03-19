import logging
import numpy as np
from configs.channel_config import SpectroscopyChannel
from domain.star_catalog import StarCatalog
from instrument.spectrum_spread import get_spectrum_placement, smear_1d_spectrum_dispersion, spread_1d_spectrum_to_2d


def generate_background_star_spectroscopy_image(channel: SpectroscopyChannel, background_stars_catalog: StarCatalog, roll_angle_start: float, roll_angle_stop: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_bands: dict[str, dict[str, float]] = {}

    spectrum_placement = get_spectrum_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_in_slit = 0
    star_ids_in_slit: list[str] = []
    star_exposure_s_by_id: dict[str, float] = {}
    is_vis_channel = channel.channel_name.upper() == "VIS"
    for star_id in background_stars_catalog.stars_by_id:
        bg_star_result = _render_star_if_in_slit(star_id, channel, background_stars_catalog, frame_index, spectrum_placement, roll_angle_start, roll_angle_stop)
        if bg_star_result is not None:
            bg_star_2d, y_positions, rendered_exposure_s = bg_star_result
            image += bg_star_2d
            n_in_slit += 1
            star_ids_in_slit.append(star_id)
            star_exposure_s_by_id[star_id] = float(rendered_exposure_s)
            if is_vis_channel and len(y_positions) > 0:
                background_star_bands[star_id] = {
                    "y0": float(np.mean(y_positions)),
                    "sigma": float(max(channel.spread_half_height_pix, (max(y_positions) - min(y_positions)) / 2.0)),
                }

    star_ids_mag_in_slit: list[str] = []
    star_ids_mag_texp_in_slit: list[str] = []
    for sid in star_ids_in_slit:
        mag = background_stars_catalog.stars_by_id[sid].gaia_magnitude
        mag_val = mag if mag is not None else float("nan")
        star_ids_mag_in_slit.append(f"{sid}:{mag_val:.3f}")
        t_exp_s = star_exposure_s_by_id.get(sid, 0.0)
        star_ids_mag_texp_in_slit.append(f"{sid}:{mag_val:.3f}:{t_exp_s:.3f}s")

    img_sum = float(np.sum(image))
    img_max = float(np.max(image)) if image.size > 0 else 0.0
    logging.info("BG STARS Slit and rendering in frame with roll_angle: frame=%d channel=%s roll_angle_start=%g roll_angle_stop=%g n_in_slit=%d/%d image_sum=%g image_max=%g star_ids_mag_texp_in_slit=[%s]", frame_index, channel.channel_name, float(roll_angle_start), float(roll_angle_stop), int(n_in_slit), int(total), img_sum, img_max, ", ".join(star_ids_mag_texp_in_slit))

    return image, background_star_bands


def _render_star_if_in_slit(star_id: str, channel: SpectroscopyChannel, catalog: StarCatalog, frame_index: int, spectrum_placement: tuple[int, float, float, float], roll_angle_start: float, roll_angle_stop: float) -> tuple[np.ndarray, list[int], float] | None:
    """Return 2d image, sampled detector rows, and rendered exposure in seconds."""
    x_target, y_target, slope, intercept = spectrum_placement
    dx, dy = catalog.get_offset_arcsec(star_id)

    counts_s_px = _get_cached_counts(star_id, catalog, channel, frame_index)
    if counts_s_px is None:
        return None
        
    roll_angles = _compute_roll_angle_samples(dx, dy, channel, roll_angle_start, roll_angle_stop)
    dt_per_sample = channel.exposure_s / float(len(roll_angles))
    star_image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    valid_y_positions: list[int] = []

    for roll_angle_deg in roll_angles:
        slit = _build_slit(channel, roll_angle_deg)
        cos_roll, sin_roll, half_w, half_l = slit
        uv = _slit_check(dx, dy, cos_roll, sin_roll, half_w, half_l)

        if uv is None:
            continue

        _, v = uv
        y_row = _detector_row(y_target, v, channel)
        counts_this_step = counts_s_px * dt_per_sample
        img = _render_spectrum_to_2d(counts_this_step, channel, x_target, y_row, slope, intercept)
        star_image += img
        valid_y_positions.append(y_row)

    if len(valid_y_positions) == 0:
        return None

    rendered_exposure_s = len(valid_y_positions) * dt_per_sample
    return star_image, valid_y_positions, rendered_exposure_s


def _build_slit(channel: SpectroscopyChannel, roll_angle_deg: float) -> tuple[float, float, float, float]:
    rad = np.deg2rad(float(roll_angle_deg))
    cos_roll = float(np.cos(rad))
    sin_roll = float(np.sin(rad))
    half_w = float(channel.slit_half_width_arcsec)
    half_l = float(channel.slit_half_length_arcsec)
    return cos_roll, sin_roll, half_w, half_l


def _slit_check(dx: float, dy: float, cos_roll: float, sin_roll: float, half_w: float, half_l: float):
    u = dx * cos_roll + dy * sin_roll
    v = -dx * sin_roll + dy * cos_roll
    if abs(u) > half_w or abs(v) > half_l:
        return None
    return [u, v]


def _get_cached_counts(star_id: str, catalog: StarCatalog, channel: SpectroscopyChannel, frame_index: int) -> np.ndarray | None:
    key = (star_id, channel.channel_name)
    if key not in catalog.counts_by_id_and_band:
        logging.info("BG STAR missing cached counts: frame=%d channel=%s star_id=%s", frame_index, channel.channel_name, star_id)
        return None
    return catalog.counts_by_id_and_band[key]


def _detector_row(y_target: float, v_arcsec: float, channel: SpectroscopyChannel) -> int:
    y_offset_pix = v_arcsec / channel.pixel_scale
    y_row = int(round(y_target + y_offset_pix))
    logging.info("BG star row placement: target star (y0) y position: %.0f, background star y position: %d", y_target, y_row)
    return y_row


def _render_spectrum_to_2d(counts_px: np.ndarray, channel: SpectroscopyChannel, x_target: int, y_row: int, slope: float, intercept: float) -> np.ndarray:
    counts_smeared = smear_1d_spectrum_dispersion(counts_px, channel)
    return spread_1d_spectrum_to_2d(counts_smeared, channel, x_target, float(y_row), slope, intercept, announce_user=False)


def _compute_roll_angle_samples(dx: float, dy: float, channel: SpectroscopyChannel, roll_angle_start: float, roll_angle_stop: float, max_motion_per_step_px: float = 0.25) -> np.ndarray:
    radius_arcsec = float(np.hypot(dx, dy))
    delta_angle_rad = float(abs(np.deg2rad(roll_angle_stop - roll_angle_start)))
    arc_length_arcsec = radius_arcsec * delta_angle_rad
    arc_length_px = arc_length_arcsec / float(channel.pixel_scale)
    n_steps = max(2, int(np.ceil(arc_length_px / max_motion_per_step_px)) + 1)
    roll_angles = np.linspace(roll_angle_start, roll_angle_stop, n_steps, dtype=np.float32)
    return roll_angles
