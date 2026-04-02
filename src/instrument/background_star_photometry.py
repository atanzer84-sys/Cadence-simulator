import numpy as np
from configs.channel_config import PhotometryChannel
from domain.star_catalog import StarCatalog
from instrument.psf_spread import get_photometry_placement, paste_psf_stamp
import logging
from instrument.background_star_common import compute_roll_angle_samples, get_cached_counts, check_within_rotated_bounds, build_rotated_bounds

_LAST_VISIBLE_SIGNATURE_BY_CHANNEL: dict[str, tuple[str, ...]] = {}

def generate_background_star_photometry_image(channel: PhotometryChannel, background_stars_catalog: StarCatalog, roll_angle_start: float, roll_angle_stop: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_star_arcs: dict[str, list[tuple[int, int]]] = {}
    
    target_star_placement = get_photometry_placement(channel)
    total = len(background_stars_catalog.stars_by_id)
    n_on_detector = 0
    stars_on_detector_ids = []

    for star_id in background_stars_catalog.stars_by_id:
        valid_positions = _render_star_if_on_detector(star_id, channel, background_stars_catalog, frame_index, target_star_placement, roll_angle_start, roll_angle_stop, image)
        
        if valid_positions is None:
            continue

        background_star_arcs[star_id] = valid_positions
        n_on_detector += 1

        # Collect ID + magnitude info for logging (similar to spectroscopy helper).
        stars_on_detector_ids.append(star_id)

    _log_background_stars_on_detector(frame_index, channel.channel_name, roll_angle_start, roll_angle_stop, n_on_detector, total, stars_on_detector_ids, background_stars_catalog, background_star_arcs)

    return image, background_star_arcs

def _render_star_if_on_detector(star_id: str, channel: PhotometryChannel, catalog: StarCatalog, frame_index: int, target_star_placement: tuple[int, float], roll_angle_start, roll_angle_stop, image: np.ndarray) -> list[tuple[int, int]] | None:

    x_target, y_target = target_star_placement
    dx, dy = catalog.get_offset_arcsec(star_id)
    separation = catalog.get_separation_arcsec(star_id)

    detector_half_bounds = (float(channel.x_pixels * channel.pixel_scale * 0.5), float(channel.y_pixels * channel.pixel_scale * 0.5))

    total_flux_electrons_per_second = get_cached_counts(star_id, catalog, channel, frame_index)
    if total_flux_electrons_per_second is None:
        return None

    roll_angles = compute_roll_angle_samples(separation, channel, roll_angle_start, roll_angle_stop)
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


    psf_stamp = channel.psf_image * flux_per_step_electrons_per_second
    for x_background_star, y_background_star in valid_positions:
        paste_psf_stamp(image, psf_stamp, x_background_star, y_background_star, psf_center_x, psf_center_y)

    return valid_positions


def _detector_position(x_target: int, y_target: int, u: float, v: float, channel: PhotometryChannel) -> tuple[int, int]:
    x_background_star = int(round(x_target + u / channel.pixel_scale))
    y_background_star = int(round(y_target + v / channel.pixel_scale))
    return x_background_star, y_background_star

def _log_background_stars_on_detector(frame_index: int, channel_name: str, roll_angle_start: float, roll_angle_stop: float, n_on_detector: int, total: int, stars_on_detector_ids: list[str], background_stars_catalog: StarCatalog, background_star_arcs: dict[str, list[tuple[int, int]]]) -> None:
    current_signature = tuple(sorted(stars_on_detector_ids))
    previous_signature = _LAST_VISIBLE_SIGNATURE_BY_CHANNEL.get(channel_name)

    if frame_index == 0 or previous_signature is None:
        full_list = []
        for star_id in stars_on_detector_ids:
            bg_star = background_stars_catalog.stars_by_id[star_id]
            valid_positions = background_star_arcs[star_id]
            x_positions = [pos[0] for pos in valid_positions]
            y_positions = [pos[1] for pos in valid_positions]
            full_list.append(f"ID={star_id} | G={bg_star.gaia_magnitude} | XCOL={min(x_positions)}-{max(x_positions)} | YROW={min(y_positions)}-{max(y_positions)}")
        logging.info("Background Stars on Detector, rendering frame with roll_angle: frame=%d channel=%s roll_angle_start=%g roll_angle_stop=%g n_on_detector=%d/%d stars=[%s]", frame_index, channel_name, float(roll_angle_start), float(roll_angle_stop), int(n_on_detector), int(total), " ; ".join(full_list))
    else:
        prev_ids = set(previous_signature)
        curr_ids = set(current_signature)

        added_ids = curr_ids - prev_ids
        removed_ids = prev_ids - curr_ids

        if added_ids or removed_ids:
            added_list = []
            for star_id in added_ids:
                bg_star = background_stars_catalog.stars_by_id[star_id]
                valid_positions = background_star_arcs[star_id]
                x_positions = [pos[0] for pos in valid_positions]
                y_positions = [pos[1] for pos in valid_positions]
                added_list.append(f"ID={star_id} | G={bg_star.gaia_magnitude} | XCOL={min(x_positions)}-{max(x_positions)} | YROW={min(y_positions)}-{max(y_positions)}")

            removed_list = []
            for star_id in removed_ids:
                removed_list.append(f"ID={star_id}")

            logging.info("Background Stars on Detector, rendering frame with roll_angle: frame=%d channel=%s roll_angle_start=%g roll_angle_stop=%g n_on_detector=%d/%d added=[%s] removed=[%s]", frame_index, channel_name, float(roll_angle_start), float(roll_angle_stop), int(n_on_detector), int(total), " ; ".join(added_list), " ; ".join(removed_list))

    _LAST_VISIBLE_SIGNATURE_BY_CHANNEL[channel_name] = current_signature

def photometry_radius_arcsec(channel: PhotometryChannel) -> float:
    half_width_arcsec = 0.5 * float(channel.x_pixels) * float(channel.pixel_scale)
    half_height_arcsec = 0.5 * float(channel.y_pixels) * float(channel.pixel_scale)
    return (half_width_arcsec * half_width_arcsec + half_height_arcsec * half_height_arcsec) ** 0.5
