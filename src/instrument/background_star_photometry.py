import numpy as np
from configs.channel_config import PhotometryChannel
from domain.star_catalog import StarCatalog
from instrument.psf_spread import get_photometry_placement, paste_psf_stamp
import logging
from instrument.background_star_common import compute_roll_angle_samples, get_cached_counts, check_within_rotated_bounds, build_rotated_bounds

def generate_background_star_photometry_image(channel: PhotometryChannel, background_stars_catalog: StarCatalog, roll_angle_start: float, roll_angle_stop: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_arcs: dict[str, list[tuple[int, int]]] = {}
    
    target_star_placement = get_photometry_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_on_detector = 0
    star_infos: list[str] = []

    for star_id in background_stars_catalog.stars_by_id:
        bg_star_result = _render_star_if_on_detector(star_id, channel, background_stars_catalog, frame_index, target_star_placement, roll_angle_start, roll_angle_stop)
        
        if bg_star_result is None:
            continue

        bg_star_image, valid_positions = bg_star_result
        image += bg_star_image
        background_star_arcs[star_id] = valid_positions
        n_on_detector += 1

        # Collect ID + magnitude info for logging (similar to spectroscopy helper).
        bg_star = background_stars_catalog.stars_by_id[star_id]

        formatted = f"{int(star_id.split('_')[1]):,}".replace(",", " ")
        mag = bg_star.gaia_magnitude if bg_star.gaia_magnitude is not None else float("nan")
        star_infos.append(f"{formatted}(id={star_id},mag={mag:.3f})")

    star_ids_on_frame = ",".join(star_infos)
    exposure_s = getattr(channel, "exposure_s", None)
    logging.info("BG STARS ARC: frame=%d channel=%s exptime_s=%s roll_angle_start=%g roll_angle_stop=%g n_on_detector=%d/%d star_ids_on_frame=%s", frame_index, channel.channel_name, str(exposure_s), float(roll_angle_start), float(roll_angle_stop), int(n_on_detector), int(total), star_ids_on_frame)


    return image, background_star_arcs

def _render_star_if_on_detector(star_id: str, channel: PhotometryChannel, catalog: StarCatalog, frame_index: int, target_star_placement: tuple[int, float], roll_angle_start, roll_angle_stop) -> tuple[np.ndarray, list[tuple[int, int]]] | None:

    x_target, y_target = target_star_placement
    dx, dy = catalog.get_offset_arcsec(star_id)
    detector_half_bounds = (float(channel.x_pixels * channel.pixel_scale * 0.5), float(channel.y_pixels * channel.pixel_scale * 0.5))

    total_flux_electrons_per_second = get_cached_counts(star_id, catalog, channel, frame_index)
    if total_flux_electrons_per_second is None:
        return None

    roll_angles = compute_roll_angle_samples(dx, dy, channel, roll_angle_start, roll_angle_stop)
    star_image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    valid_positions: list[tuple[int, int]] = []

    for roll_angle_deg in roll_angles:
        detector = build_rotated_bounds(detector_half_bounds, roll_angle_deg)
        uv = check_within_rotated_bounds(dx, dy, detector)

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

    logging.info("BG STAR START: frame=%d channel=%s star_id=%s dx=%.6f dy=%.6f n_positions=%d total_flux_e_s=%.6e flux_per_step_e_s=%.6e first_pos=%s last_pos=%s", frame_index, channel.channel_name, star_id, dx, dy, len(valid_positions), total_flux_electrons_per_second, flux_per_step_electrons_per_second, valid_positions[0], valid_positions[-1])

    for i, (x_background_star, y_background_star) in enumerate(valid_positions):

        psf_stamp = channel.psf_image * flux_per_step_electrons_per_second

        stamp_sum_before = float(np.sum(psf_stamp))
        stamp_max_before = float(np.max(psf_stamp))
        before_sum = float(np.sum(star_image))

        paste_psf_stamp(star_image, psf_stamp, x_background_star, y_background_star, psf_center_x, psf_center_y)
        after_sum = float(np.sum(star_image))
        pasted_delta = after_sum - before_sum

        logging.info("BG STAR STEP: frame=%d channel=%s star_id=%s step=%d/%d det_x=%d det_y=%d stamp_sum=%.6e stamp_max=%.6e pasted_delta=%.6e", frame_index, channel.channel_name, star_id, i + 1, len (valid_positions), x_background_star, y_background_star, stamp_sum_before, stamp_max_before, pasted_delta)
    final_sum = float(np.sum(star_image))

    final_max = float(np.max(star_image))
    nonzero_y, nonzero_x = np.nonzero(star_image)

    if nonzero_y.size > 0:
        bbox = f"y={int(np.min(nonzero_y))}..{int(np.max(nonzero_y))}, x={int(np.min(nonzero_x))}..{int(np.max(nonzero_x))}"
    else:
        bbox = "none"

    logging.info("BG STAR FINAL: frame=%d channel=%s star_id=%s final_sum=%.6e final_max=%.6e bbox=%s", frame_index, channel.channel_name, star_id, final_sum, final_max, bbox)

    return star_image, valid_positions


def _detector_position(x_target: int, y_target: int, u: float, v: float, channel: PhotometryChannel) -> tuple[int, int]:
    x_background_star = int(round(x_target + u / channel.pixel_scale))
    y_background_star = int(round(y_target + v / channel.pixel_scale))
    return x_background_star, y_background_star



