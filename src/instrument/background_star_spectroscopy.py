import logging
import numpy as np
from configs.channel_config import SpectroscopyChannel
from domain.star_catalog import StarCatalog
from instrument.spectrum_spread import get_spectrum_placement, smear_1d_spectrum_dispersion, spread_1d_spectrum_to_2d
from instrument.background_star_common import compute_roll_angle_samples, get_cached_counts, rotated_coordinates, within_rotated_bounds_mask

type StarInSlitLog = dict[str, str | float | int]


def generate_background_star_spectroscopy_image(channel: SpectroscopyChannel, background_stars_catalog: StarCatalog, roll_angle_start: float, roll_angle_stop: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_bands: dict[str, dict[str, float]] = {}

    spectrum_placement = get_spectrum_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_in_slit = 0
    stars_in_slit_log: list[StarInSlitLog] = []
    is_vis_channel = channel.channel_name.upper() == "VIS"
    slit_reach_arcsec = spectroscopy_radius_arcsec(channel)
    
    for star_id in background_stars_catalog.stars_by_id:
        separation_arcsec = background_stars_catalog.get_separation_arcsec(star_id)

        if separation_arcsec > slit_reach_arcsec:
            continue

        bg_star_result = _render_star_if_in_slit(star_id, image, channel, background_stars_catalog, frame_index, spectrum_placement, roll_angle_start, roll_angle_stop)

        if bg_star_result is not None:
            y_positions, rendered_exposure_s = bg_star_result
        
            n_in_slit += 1
            stars_in_slit_log.append(
                {
                    "id": star_id,
                    "mag": background_stars_catalog.stars_by_id[star_id].gaia_magnitude,
                    "exp_s": float(rendered_exposure_s),
                    "yrow_min": int(min(y_positions)),
                    "yrow_max": int(max(y_positions)),
                }
            )
            if is_vis_channel and len(y_positions) > 0:
                background_star_bands[star_id] = {
                    "y0": float(np.mean(y_positions)),
                    "sigma": float(max(channel.spread_half_height_pix, (max(y_positions) - min(y_positions)) / 2.0)),
                }

    _log_background_stars_in_slit(frame_index, channel.channel_name, roll_angle_start, roll_angle_stop, n_in_slit, total, stars_in_slit_log)

    return image, background_star_bands


def _render_star_if_in_slit(star_id: str, image, channel: SpectroscopyChannel, catalog: StarCatalog, frame_index: int, spectrum_placement: tuple[int, float, float, float], roll_angle_start: float, roll_angle_stop: float) -> tuple[list[int], float] | None:
    """Render directly into star_image and return sampled detector rows and rendered exposure in seconds."""
    x_target, y_target, slope, intercept = spectrum_placement
    dx, dy = catalog.get_offset_arcsec(star_id)
    separation = catalog.get_separation_arcsec(star_id)

    slit_half_bounds = (float(channel.slit_half_width_arcsec), float(channel.slit_half_length_arcsec))

    counts_s_px = get_cached_counts(star_id, catalog, channel, frame_index)
    if counts_s_px is None:
        return None
        
    roll_angles = compute_roll_angle_samples(separation, channel, roll_angle_start, roll_angle_stop)
    dt_per_sample = channel.exposure_s / float(len(roll_angles))
    valid_y_positions: list[int] = []

    # we want to remember the time per y-row. spreading is super expensive and we don't need to spread if we already spread one specific y-row
    time_per_row: dict[int, float] = {}

    u, v = rotated_coordinates(dx, dy, roll_angles)
    visible = within_rotated_bounds_mask(u, v, slit_half_bounds)

    if not np.any(visible):
        return None

    for v_arcsec in v[visible]:
        y_row = _detector_row(y_target, float(v_arcsec), channel)
        valid_y_positions.append(y_row)

        if y_row in time_per_row:
            time_per_row[y_row] += dt_per_sample
        else:
            time_per_row[y_row] = dt_per_sample
            
    if len(valid_y_positions) == 0:
        return None

    counts_smeared_per_second = smear_1d_spectrum_dispersion(counts_s_px, channel)

    for y_row, total_time in time_per_row.items():
        counts_this_step = counts_smeared_per_second * total_time

        spread_1d_spectrum_to_2d(image, counts_this_step, channel, (x_target, float(y_row), slope, intercept), announce_user=False)


    rendered_exposure_s = len(valid_y_positions) * dt_per_sample
    return valid_y_positions, rendered_exposure_s


def _detector_row(y_target: float, v_arcsec: float, channel: SpectroscopyChannel) -> int:
    y_offset_pix = v_arcsec / channel.pixel_scale
    y_row = int(round(y_target + y_offset_pix))
    return y_row


def _log_background_stars_in_slit(frame_index: int, channel_name: str, roll_angle_start: float, roll_angle_stop: float, n_in_slit: int, total: int, stars_in_slit_log: list[StarInSlitLog]) -> None:
    star_ids_mag_texp_in_slit: list[str] = []
    for star_info in stars_in_slit_log:
        star_ids_mag_texp_in_slit.append(f"ID={star_info['id']} | G={star_info['mag']} | EXP={star_info['exp_s']:.3f}s | YROW={star_info['yrow_min']}-{star_info['yrow_max']}")
    logging.info("Background Stars in Slit, rendering frame with roll_angle: frame=%d channel=%s roll_angle_start=%g roll_angle_stop=%g n_in_slit=%d/%d star_ids_mag_texp_in_slit=[%s]", frame_index, channel_name, float(roll_angle_start), float(roll_angle_stop), int(n_in_slit), int(total), " ; ".join(star_ids_mag_texp_in_slit))


def spectroscopy_radius_arcsec(channel: SpectroscopyChannel) -> float:
    x = float(channel.slit_half_width_arcsec)
    y = float(channel.slit_half_length_arcsec)
    return (x * x + y * y) ** 0.5
