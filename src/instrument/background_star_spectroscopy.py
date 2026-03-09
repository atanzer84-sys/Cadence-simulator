import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
from domain.star_catalog import StarCatalog
from instrument.spectrum_spread import get_spectrum_placement, smear_1d_spectrum_dispersion, spread_1d_spectrum_to_2d


def generate_background_star_spectroscopy_image(channel: SpectroscopyChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog, roll_angle_deg: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_bands: dict[str, dict[str, float]] = {}

    slit = _build_slit(channel, roll_angle_deg)
    spectrum_placement = get_spectrum_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_in_slit = 0
    is_vis_channel = channel.channel_name.upper() == "VIS"
    for star_id in background_stars_catalog.stars_by_id:
        bg_star_result = _render_star_if_in_slit(star_id, channel, background_stars_catalog, frame_index, slit, spectrum_placement)
        if bg_star_result is not None:
            bg_star_2d, y0_background_star = bg_star_result
            image += bg_star_2d
            n_in_slit += 1
            if is_vis_channel:
                background_star_bands[star_id] = {
                    "y0": float(y0_background_star),
                    "sigma": float(channel.spread_half_height_pix),
                }
    logging.info("BG STARS roll_angle: frame=%d channel=%s roll_angle_deg=%g n_in_slit=%d/%d image_sum=%g", frame_index, channel.channel_name, float(roll_angle_deg), int(n_in_slit), int(total), float(np.sum(image)))
    return image, background_star_bands


def _render_star_if_in_slit(star_id: str, channel: SpectroscopyChannel, catalog: StarCatalog, frame_index: int, slit: tuple[float, float, float, float], spectrum_placement: tuple[int, float, float, float]) -> tuple[np.ndarray, int] | None:
    """Return 2d image for star if inside slit and has counts, else None."""
    cos_roll, sin_roll, half_w, half_l = slit
    x_target, y_target, slope, intercept = spectrum_placement
    dx, dy = catalog.get_offset_arcsec(star_id)
    uv = _slit_check(dx, dy, cos_roll, sin_roll, half_w, half_l)
    if uv is None:
        return None
    u, v = uv
    counts = _get_cached_counts(star_id, catalog, channel, frame_index)
    if counts is None:
        return None
    y_row = _detector_row(y_target, v, channel)
    img = _render_spectrum_to_2d(counts, channel, x_target, y_row, slope, intercept)
    _log_star_in_slit(star_id, catalog, channel, frame_index, dx, dy, u, v, half_w, half_l, y_row, x_target)
    return img, y_row


def _build_slit(channel: SpectroscopyChannel, roll_angle_deg: float) -> tuple[float, float, float, float]:
    rad = np.deg2rad(float(roll_angle_deg))
    cos_roll = float(np.cos(rad))
    sin_roll = float(np.sin(rad))
    half_w = float(channel.slit_half_width_arcsec)
    half_l = float(channel.slit_half_length_arcsec)
    return cos_roll, sin_roll, half_w, half_l


def _slit_check(dx: float, dy: float, cos_roll: float, sin_roll: float, half_w: float, half_l: float):
    """Return [u, v] if inside slit, else None."""
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


def _render_spectrum_to_2d(counts_s_px: np.ndarray, channel: SpectroscopyChannel, x_target: int, y_row: int, slope: float, intercept: float) -> np.ndarray:
    counts_smeared = smear_1d_spectrum_dispersion(counts_s_px, channel)
    return spread_1d_spectrum_to_2d(counts_smeared, channel, x_target, float(y_row), slope, intercept, announce_user=False)


def _log_star_in_slit(star_id: str, catalog: StarCatalog, channel: SpectroscopyChannel, frame_index: int, dx: float, dy: float, u: float, v: float, half_w: float, half_l: float, detector_y: float, x_target: int) -> None:
    bg_star = catalog.stars_by_id[star_id]
    formatted = f"{int(star_id.split('_')[1]):,}".replace(",", " ")
    mag = bg_star.gaia_magnitude if bg_star.gaia_magnitude is not None else float("nan")
    ra = bg_star.right_ascension if bg_star.right_ascension is not None else float("nan")
    dec = bg_star.declination if bg_star.declination is not None else float("nan")
    logging.info("BG STAR in slit: frame=%d channel=%s star_id_formatted=%s star_id=%s mag=%.3f dx=%g dy=%g u=%g v=%g half_w=%g half_l=%g ra=%.6f dec=%.6f detector_y=%.2f x=%d", frame_index, channel.channel_name, formatted, star_id, mag, dx, dy, u, v, half_w, half_l, ra, dec, detector_y, x_target)
