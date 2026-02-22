"""Spread 1D spectrum to 2D detector image (Gaussian or wavelength-dependent profile)."""
import numpy as np
import logging
from configs.channel_config import SpectroscopyChannel


def spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel: SpectroscopyChannel, header=None):

    logging.info("Building single Science Frame: channel=%s.", channel.channel_name)

    nx = channel.x_pixels
    mode = channel.mode

    if len(counts_s_pixel_convolved) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s counts_len=%d nx=%d", channel.channel_name, int(len(counts_s_pixel_convolved)), int(nx))
        print(f"PROFILE SPREAD ERROR: channel={channel.channel_name} counts_len={len(counts_s_pixel_convolved)} nx={nx}")
        raise ValueError(f"Counts length {len(counts_s_pixel_convolved)} does not match nx {nx}")

    # no lookup or high resolution spectrograph spreading as of now.
    if mode == 1:

        if (
            channel.spread_y_positions is not None
            and channel.spread_y_weights is not None
            and channel.spread_y_wavelengths is not None
        ):
            # wavelength dependent spreading will be used
            return _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel, header=header)

        else:
            # Gaussian dependent spreading
            return _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel, header=header)

    msg = f"mode={mode} not implemented yet (only mode=1 is supported)"
    logging.error(msg)
    raise NotImplementedError(msg)


def _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel: SpectroscopyChannel, header=None):

    nx = channel.x_pixels
    ny = channel.y_pixels
    spread_half_height = channel.spread_half_height_pix

    if spread_half_height <= 0:
        logging.error("SPREAD CONFIG ERROR: channel=%s no spread profile and spread_half_height_pix=%d", channel.channel_name, channel.spread_half_height_pix)
        raise ValueError("No cross-dispersion spreading configured")

    # choose a fixed trace center (placeholder)
    y0 = ny // 2
    spatial_sigma_pix = float(channel.spread_half_height_pix)

    # build a normalized vertical profile w[y] with sum(w)=1
    w = np.zeros(ny, dtype=np.float64)
    for y in range(ny):
        dy = y - y0
        w[y] = np.exp(-0.5 * (dy / spatial_sigma_pix) * (dy / spatial_sigma_pix))

    w_sum = w.sum()
    if w_sum <= 0.0:
        logging.info("vertical profile w_sum=%g, normalizing", w_sum)
        raise ValueError("vertical profile sum <= 0")
    for y in range(ny):
        w[y] = w[y] / w_sum

    # fill the 2D image: for each x-column, distribute counts[x] over y using w[y]
    image = np.zeros((ny, nx), dtype=np.float64)
    for x in range(nx):
        for y in range(ny):
            image[y, x] = counts_s_pixel_convolved[x] * w[y]

    logging.info("GAUSSIAN SPREAD RESULT: channel=%s shape=(%d,%d) sum=%g", channel.channel_name, image.shape[0], image.shape[1], float(image.sum()))
    col_sums = image.sum(axis=0)
    logging.info("GAUSSIAN SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))

    if not np.allclose(col_sums, counts_s_pixel_convolved, rtol=1e-10, atol=1e-12):
        logging.error("GAUSSIAN SPREAD CHECK FAILED: channel=%s column sums do not match input counts", channel.channel_name)
        raise ValueError("Gaussian spread column sum mismatch")

    return image, header


def _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel: SpectroscopyChannel, header=None):
    logging.info("WAVELENGTH DEPENDENT SPREAD: channel=%s spread_file=%s mode=1 profile detected but not yet implemented", channel.channel_name, channel.spread_profile_file)

    nx = channel.x_pixels
    ny = channel.y_pixels

    spread_y_pos = channel.spread_y_positions
    spread_weigths = channel.spread_y_weights
    spread_wavelengths = channel.spread_y_wavelengths
    detector_wavelengths = channel.wavelength

    if len(detector_wavelengths) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s detector wavelength length mismatch", channel.channel_name)
        raise ValueError("Detector wavelength grid length mismatch")

    dy = np.round(spread_y_pos).astype(np.int64)
    y0 = ny // 2

    logging.info("PROFILE SPREAD START: channel=%s spread_file=%s nx=%d ny=%d n_bins=%d y0=%d", channel.channel_name, channel.spread_profile_file, int(nx), int(ny), int(spread_wavelengths.shape[0]), int(y0))

    image = np.zeros((ny, nx), dtype=np.float64)

    for x in range(nx):
        lam = float(detector_wavelengths[x])
        j = int(np.argmin(np.abs(spread_wavelengths - lam)))

        c = float(counts_s_pixel_convolved[x])
        for i in range(dy.shape[0]):
            y = int(y0 + dy[i])
            if 0 <= y < ny:
                image[y, x] += c * float(spread_weigths[i, j])

    col_sums = image.sum(axis=0)
    logging.info("PROFILE SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))

    return image, header
