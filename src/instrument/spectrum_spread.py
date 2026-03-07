"""Spread 1D spectrum to 2D detector image (Gaussian or wavelength-dependent profile)."""
import numpy as np
import logging

from configs.channel_config import SpectroscopyChannel
from utils.helpers import announce


def get_spectrum_placement(channel: SpectroscopyChannel) -> tuple[int, float, float, float]:
    """Return (x0, y0, slope, intercept_pixels) for target star. For background stars: use x0, slope, intercept; supply y0 per star."""
    x0, y0 = get_target_star_detector_position(channel)
    slope, intercept = channel.slope, channel.intercept_pixels
    if slope != 0.0 or intercept != 0.0:
        logging.error("SPREAD PLACEMENT ERROR: channel=%s slope=%g intercept=%g not supported", channel.channel_name, slope, intercept)
        raise ValueError("slope and intercept_pixels must be 0 (not yet supported)")
    return x0, float(y0), slope, intercept


def smear_1d_spectrum_dispersion(counts_s_px: np.ndarray, channel: SpectroscopyChannel) -> np.ndarray:
    """Apply charge-transfer smear along dispersion axis. Returns counts/s/pixel scaled by crossing_time_s."""
    n_smear_steps = int(round(channel.smear_shift_pixels))
    if n_smear_steps <= 0:
        return counts_s_px * channel.crossing_time_s
    counts_smeared_px = np.zeros_like(counts_s_px)
    counts_smear_step_px = counts_s_px * (channel.crossing_time_s / n_smear_steps)
    half = n_smear_steps // 2
    for x_shift in range(-half, -half + n_smear_steps):
        if x_shift == 0:
            counts_smeared_px += counts_smear_step_px
        elif x_shift > 0:
            counts_smeared_px[x_shift:] += counts_smear_step_px[:-x_shift]
        else:
            counts_smeared_px[:x_shift] += counts_smear_step_px[-x_shift:]
    return counts_smeared_px


def spread_target_star_spectrum_to_2d(counts_s_pixel_convolved, channel: SpectroscopyChannel):
    """Entry point for target star: derive spatial info and spread 1D spectrum to 2D."""
    x0, y0, slope, intercept = get_spectrum_placement(channel)
    return spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel, x0, y0, slope, intercept)


def spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel: SpectroscopyChannel, x0: int, y0: float, slope: float, intercept: float, announce_user: bool = True):
    """Spread 1D spectrum to 2D. Caller provides placement (x0, y0, slope, intercept). For background stars: x0, _, slope, intercept = get_spectrum_placement(channel); y0 = y_background_star."""
    announce(f"Spreading 1D counts to 2D detector image for channel {channel.channel_name}.", to_user=announce_user)

    nx = channel.x_pixels
    mode = channel.mode

    if len(counts_s_pixel_convolved) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s counts_len=%d nx=%d", channel.channel_name, int(len(counts_s_pixel_convolved)), int(nx))
        print(f"PROFILE SPREAD ERROR: channel={channel.channel_name} counts_len={len(counts_s_pixel_convolved)} nx={nx}")
        raise ValueError(f"Counts length {len(counts_s_pixel_convolved)} does not match nx {nx}")

    # no lookup or high resolution spectrograph spreading as of now.
    if mode != 1:
        msg = f"mode={mode} not implemented yet (only mode=1 is supported)"
        logging.error(msg)
        raise NotImplementedError(msg)

    if (
        channel.spread_y_positions is not None
        and channel.spread_y_weights is not None
        and channel.spread_y_wavelengths is not None
    ):
        return _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel, x0, y0, slope, intercept)
    else:
        return _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel, x0, y0, slope, intercept)



def _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel: SpectroscopyChannel, x0: int, y0: float, slope: float, intercept: float):
    nx = channel.x_pixels
    ny = channel.y_pixels
    spread_half_height = channel.spread_half_height_pix

    if spread_half_height <= 0:
        logging.error("SPREAD CONFIG ERROR: channel=%s no spread profile and spread_half_height_pix=%d", channel.channel_name, channel.spread_half_height_pix)
        raise ValueError("No cross-dispersion spreading configured")

    spatial_sigma_pix = float(channel.spread_half_height_pix)

    # case if slope and intercept are 0, then this doesn't have to be calculated in a for loop - performance increase!
    if slope == 0.0 and intercept == 0.0:
        w = _gaussian_vertical_profile(ny, y0, spatial_sigma_pix)
        image = np.outer(w, counts_s_pixel_convolved)
    else:
        image = np.zeros((ny, nx), dtype=np.float64)
        for i in range(nx):
            x = x0 + i
            if 0 <= x < nx:
                y_center = y0 + intercept + slope * (x - x0)
                w = _gaussian_vertical_profile(ny, y_center, spatial_sigma_pix)
                image[:, x] = counts_s_pixel_convolved[i] * w

    logging.info("GAUSSIAN SPREAD RESULT: channel=%s shape=(%d,%d) sum=%g", channel.channel_name, image.shape[0], image.shape[1], float(image.sum()))
    col_sums = image.sum(axis=0)
    logging.info("GAUSSIAN SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))

    if not np.allclose(col_sums, counts_s_pixel_convolved, rtol=1e-10, atol=1e-12):
        logging.error("GAUSSIAN SPREAD CHECK FAILED: channel=%s column sums do not match input counts", channel.channel_name)
        raise ValueError("Gaussian spread column sum mismatch")

    return image

def _gaussian_vertical_profile(ny: int, y_center: float, sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian profile along y. Returns shape (ny,)."""
    y_coords = np.arange(ny, dtype=np.float64) - y_center
    w = np.exp(-0.5 * (y_coords / sigma) ** 2)
    w_sum = w.sum()
    if w_sum <= 0.0:
        raise ValueError("vertical profile sum <= 0")
    return w / w_sum


def _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel: SpectroscopyChannel, x0: int, y0: float, slope: float, intercept: float):
    """Vectorized wavelength-dependent profile spread."""
    logging.info("WAVELENGTH DEPENDENT SPREAD: channel=%s spread_file=%s", channel.channel_name, channel.spread_profile_file)

    nx = channel.x_pixels
    ny = channel.y_pixels

    spread_y_pos = channel.spread_y_positions
    spread_weights = channel.spread_y_weights
    spread_wavelengths = channel.spread_y_wavelengths
    detector_wavelengths = channel.effective_area_wavelength

    if len(detector_wavelengths) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s detector wavelength length mismatch", channel.channel_name)
        raise ValueError("Detector wavelength grid length mismatch")

    dy = np.round(spread_y_pos).astype(np.int64)

    logging.info("PROFILE SPREAD START: channel=%s spread_file=%s nx=%d ny=%d n_bins=%d y0=%g", channel.channel_name, channel.spread_profile_file, int(nx), int(ny), int(spread_wavelengths.shape[0]), y0)

    image = np.zeros((ny, nx), dtype=np.float64)

    x_indices = np.arange(nx, dtype=np.int64)
    y_centers = y0 + intercept + slope * (x_indices - x0)
    j_indices = np.argmin(np.abs(spread_wavelengths[:, None] - detector_wavelengths[None, :]), axis=0)

    for k in range(dy.shape[0]):
        y_all = np.round(y_centers + dy[k]).astype(np.int64)
        weights = spread_weights[k, j_indices]
        values = counts_s_pixel_convolved * weights
        mask = (y_all >= 0) & (y_all < ny)
        np.add.at(image, (y_all[mask], x_indices[mask]), values[mask])

    col_sums = image.sum(axis=0)
    logging.info("PROFILE SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))

    return image


def get_target_star_detector_position(channel: SpectroscopyChannel):
    nx = channel.x_pixels
    ny = channel.y_pixels
    x0 = int(round(channel.slit_position_x_arcsec / channel.pixel_scale))
    y0 = int(round((ny // 2) + channel.slit_position_y_arcsec / channel.pixel_scale))
    logging.info("SPREAD_ANCHOR channel=%s x0=%d y0=%d nx=%d ny=%d", channel.channel_name, x0, y0, nx, ny)

    if x0 != 0:
        logging.error("SPREAD_ANCHOR_ERROR channel=%s x0=%d horizontal shift not supported", channel.channel_name, x0)
        raise ValueError("Horizontal slit_position_x_arcsec not yet supported (must be 0.0)")

    if y0 < 0 or y0 >= ny:
        logging.error("SPREAD_ANCHOR_ERROR channel=%s y0=%d ny=%d", channel.channel_name, y0, ny)
        raise ValueError("slit_position_y_arcsec places spectrum outside detector")

    return x0, y0