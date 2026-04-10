"""Spread 1D spectrum to 2D detector image (Gaussian or wavelength-dependent profile)."""
import numpy as np
import logging
from configs.channel_config import SpectroscopyChannel
from utils.helpers import announce

_PROFILE_SPREAD_WARNED_CHANNELS: set[str] = set()
_PROFILE_SPREAD_WARN_THRESHOLD = 5e-2


def spread_target_star_spectrum_to_2d(counts_s_pixel_convolved, channel: SpectroscopyChannel):
    """Entry point for target star: derive spatial info and spread 1D spectrum to 2D."""
    placement = get_spectrum_placement(channel)
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    if channel.observation_mode == "spectroscopy":
        spread_1d_spectrum_to_2d(image, counts_s_pixel_convolved, channel, placement)
        return image
    if channel.observation_mode == "spectropolarimetry":
        spread_target_star_spectropolarimetry_to_2d(image, counts_s_pixel_convolved, channel, placement)
        return image

def get_spectrum_placement(channel: SpectroscopyChannel) -> tuple[int, float, float, float]:
    """Return (x0, y0, slope, intercept_pixels) for target star. For background stars: use x0, slope, intercept; supply y0 per star."""
    x0, y0 = get_target_star_detector_position(channel)
    slope, intercept = channel.slope, channel.intercept_pixels
    if slope != 0.0 or intercept != 0.0:
        logging.error("SPREAD PLACEMENT ERROR: channel=%s slope=%g intercept=%g not supported", channel.channel_name, slope, intercept)
        raise ValueError("slope and intercept_pixels must be 0 (not yet supported)")
    return x0, float(y0), float(slope), float(intercept)

def get_target_star_detector_position(channel: SpectroscopyChannel):
    ny = channel.y_pixels
    x0 = int(round(channel.slit_position_x_arcsec / channel.pixel_scale))
    y0 = int(round((ny // 2) + channel.slit_position_y_arcsec / channel.pixel_scale))

    if x0 != 0:
        logging.error("Get target star detector position failed: channel=%s x0=%d horizontal shift not supported", channel.channel_name, x0)
        raise ValueError("Horizontal slit_position_x_arcsec not yet supported (must be 0.0)")

    if y0 < 0 or y0 >= ny:
        logging.error("Get target star detector position failed: channel=%s y0=%d ny=%d (outside detector)", channel.channel_name, y0, ny)
        raise ValueError("slit_position_y_arcsec places spectrum outside detector")

    return x0, y0

def spread_1d_spectrum_to_2d(image: np.ndarray, counts_s_pixel_convolved, channel: SpectroscopyChannel, placement, announce_user: bool = True):
    """Spread 1D spectrum to 2D. Caller provides placement (x0, y0, slope, intercept). For background stars: x0, _, slope, intercept = get_spectrum_placement(channel); y0 = y_background_star."""
    announce(f"Spreading 1D counts to 2D detector image for channel {channel.channel_name}.", to_user=announce_user)
    nx = channel.x_pixels
    mode = channel.mode

    if len(counts_s_pixel_convolved) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s counts_len=%d nx=%d", channel.channel_name, int(len(counts_s_pixel_convolved)), int(nx))
        print(f"PROFILE SPREAD ERROR: channel={channel.channel_name} counts_len={len(counts_s_pixel_convolved)} nx={nx}")
        raise ValueError(f"Counts length {len(counts_s_pixel_convolved)} does not match nx {nx}")

    # no lookup or high resolution spectrograph spreading as of now.
    # TODO: IF YOU WANT DIFFERENT MODES, IMPLEMENT MODES HERE
    if mode != 1:
        msg = f"mode={mode} not implemented yet (only mode=1 is supported)"
        logging.error(msg)
        raise NotImplementedError(msg)

    if (
        channel.spread_y_positions is not None
        and channel.spread_y_weights is not None
        and channel.spread_y_wavelengths is not None
    ):
        return _spread_1d_to_2d_profile(image, counts_s_pixel_convolved, channel, placement)
    else:
        return _spread_1d_to_2d_gaussian(image, counts_s_pixel_convolved, channel, placement)

def _spread_1d_to_2d_gaussian(image: np.ndarray, counts_s_pixel_convolved, channel: SpectroscopyChannel, placement):
    nx = channel.x_pixels
    ny = channel.y_pixels
    x0, y0, slope, intercept = placement
    spread_half_height = channel.spread_half_height_pix

    if spread_half_height <= 0:
        logging.error("SPREAD CONFIG ERROR: channel=%s no spread profile and spread_half_height_pix=%d", channel.channel_name, channel.spread_half_height_pix)
        raise ValueError("No cross-dispersion spreading configured")

    spatial_sigma_pix = float(channel.spread_half_height_pix)

    # case if slope and intercept are 0, then this doesn't have to be calculated in a for loop - performance increase!
    if slope == 0.0 and intercept == 0.0:
        counts = counts_s_pixel_convolved.astype(np.float32, copy=False)
        w = _gaussian_vertical_profile(ny, y0, spatial_sigma_pix).astype(np.float32, copy=False)
        image += np.outer(w, counts).astype(np.float32)
    else:
        for i in range(nx):
            x = x0 + i
            if 0 <= x < nx:
                y_center = y0 + intercept + slope * (x - x0)
                w = _gaussian_vertical_profile(ny, y_center, spatial_sigma_pix)
                image[:, x] += counts_s_pixel_convolved[i] * w

    
def _gaussian_vertical_profile(ny: int, y_center: float, sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian profile along y. Returns shape (ny,)."""
    y_coords = np.arange(ny, dtype=np.float64) - y_center
    w = np.exp(-0.5 * (y_coords / sigma) ** 2)
    w_sum = w.sum()
    if w_sum <= 0.0:
        raise ValueError("vertical profile sum <= 0")
    return w / w_sum

def _spread_1d_to_2d_profile(image: np.ndarray, counts_s_pixel_convolved, channel: SpectroscopyChannel, placement):
    """Vectorized wavelength-dependent profile spread."""
    nx = channel.x_pixels
    ny = channel.y_pixels
    x0, y0, slope, intercept = placement

    spread_y_pos = channel.spread_y_positions
    spread_weights = channel.spread_y_weights
    spread_wavelengths = channel.spread_y_wavelengths
    detector_wavelengths = channel.effective_area_wavelength

    if len(detector_wavelengths) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s detector wavelength length mismatch", channel.channel_name)
        raise ValueError("Detector wavelength grid length mismatch")

    dy = np.round(spread_y_pos).astype(np.int64)

    x_indices = np.arange(nx, dtype=np.int64)
    y_centers = y0 + intercept + slope * (x_indices - x0)
    j_indices = np.argmin(np.abs(spread_wavelengths[:, None] - detector_wavelengths[None, :]), axis=0)

    for k in range(dy.shape[0]):
        y_all = np.round(y_centers + dy[k]).astype(np.int64)
        weights = spread_weights[k, j_indices]
        values = counts_s_pixel_convolved * weights
        mask = (y_all >= 0) & (y_all < ny)
        np.add.at(image, (y_all[mask], x_indices[mask]), values[mask])

        
    m = float(np.max(np.abs(image.sum(axis=0) - counts_s_pixel_convolved) / np.maximum(np.abs(counts_s_pixel_convolved), 1e-30)))
    if m > _PROFILE_SPREAD_WARN_THRESHOLD and channel.channel_name not in _PROFILE_SPREAD_WARNED_CHANNELS:
        _PROFILE_SPREAD_WARNED_CHANNELS.add(channel.channel_name)
        logging.warning("PROFILE SPREAD CHECK WARN | channel=%s | max_rel_diff=%g | threshold=%g (logged once per channel)", channel.channel_name, m, _PROFILE_SPREAD_WARN_THRESHOLD)

def smear_1d_spectrum_dispersion(counts_s_px: np.ndarray, channel: SpectroscopyChannel) -> np.ndarray:
    """Apply dispersion smear while preserving total counts."""
    n_smear_steps = int(round(channel.smear_shift_pixels))

    if n_smear_steps <= 1:
        return counts_s_px.copy()

    counts_smeared_px = np.zeros_like(counts_s_px)
    counts_smear_step_px = counts_s_px / n_smear_steps

    half = n_smear_steps // 2
    for x_shift in range(-half, -half + n_smear_steps):
        if x_shift == 0:
            counts_smeared_px += counts_smear_step_px
        elif x_shift > 0:
            counts_smeared_px[x_shift:] += counts_smear_step_px[:-x_shift]
        else:
            counts_smeared_px[:x_shift] += counts_smear_step_px[-x_shift:]
    return counts_smeared_px

def spread_target_star_spectropolarimetry_to_2d(image: np.ndarray, counts_s_pixel_convolved: np.ndarray, channel: SpectroscopyChannel, placement) -> np.ndarray:
    # Split total flux into two beams using polarization delta.
    # delta is defined as normalized imbalance: delta = (beam1 - beam2) / total
    #
    # delta = 0.0  -> 50% / 50%  (unpolarized)
    # delta = 1.0  -> 100% / 0%  (fully polarized)
    #
    # Solve:
    #     beam1 + beam2 = total
    #     beam1 - beam2 = delta * total
    #
    # Result:
    #     beam1 = total * (1 + delta) / 2
    #     beam2 = total * (1 - delta) / 2

    detector_wavelength = channel.effective_area_wavelength

    delta_interp = np.interp(detector_wavelength, channel.polarization_wavelength, channel.polarization_delta)

    beam1 = counts_s_pixel_convolved * (1.0 + delta_interp) / 2.0
    beam2 = counts_s_pixel_convolved * (1.0 - delta_interp) / 2.0

    image_beam2 = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    spread_1d_spectrum_to_2d(image, beam1, channel, placement, announce_user=True)
    spread_1d_spectrum_to_2d(image_beam2, beam2, channel, placement, announce_user=False)

    logging.info("POL TEST beam split: delta[0]=%.3f delta[mid]=%.3f delta[-1]=%.3f", delta_interp[0], delta_interp[len(delta_interp) // 2], delta_interp[-1])
    logging.info("POL TEST counts: total[0]=%.3f beam1[0]=%.3f beam2[0]=%.3f", counts_s_pixel_convolved[0], beam1[0], beam2[0])
    logging.info("POL TEST counts: total[mid]=%.3f beam1[mid]=%.3f beam2[mid]=%.3f", counts_s_pixel_convolved[len(counts_s_pixel_convolved) // 2], beam1[len(beam1) // 2], beam2[len(beam2) // 2])
    logging.info("POL TEST counts: total[-1]=%.3f beam1[-1]=%.3f beam2[-1]=%.3f", counts_s_pixel_convolved[-1], beam1[-1], beam2[-1])


    separation = channel.beam_separation_pix

    image_beam2_separated = np.zeros_like(image_beam2)
    image_beam2_separated[separation:, :] = image_beam2[:-separation, :]
    
    image += image_beam2_separated
    return image
