import numpy as np
import logging
from frame.bias import generate_bias_frame
from frame.dark import generate_dark_frame

def generate_science_frames(counts_s_pixel_convolved, channel_cfg, channel_cal, n_frames, exposure_time_s, base_header):

    logging.info("Building Science Frames: n_frames=%d.", n_frames)

    ccd_gain = channel_cfg.ccd_gain

    science_frames = []
    science_headers = []

    for i in range(n_frames):
        header = base_header.copy()
        header.append(("FILETYPE", "SCIENCE",                   "Type of observation"))
        header.append(("CHANNEL", channel_cfg.channel_name,     "Detector channel"))
        header.append(("EXP_ID", f"Science {i+1}",              "Exposure ID"))
        header.append(("OBS_ID", f"Obs Science {i+1}",          "Observation ID"))
        header.append(("EXPTIME",  float(exposure_time_s),      "Exposure time of observation"))
        header.append(("CCDGAIN",  ccd_gain,                    "CCD gain"))

        # generate new detector image
        detector_image, header = spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel_cfg, channel_cal, header)

        # generate new bias and dark
        bias_frame, _ = generate_bias_frame(channel_cfg, header = None)
        dark_frame, _ = generate_dark_frame(channel_cfg, exposure_time_s, header = None)

        # combine into science
        science = (detector_image * exposure_time_s + dark_frame + bias_frame) * ccd_gain
        science_frames.append(science)
        science_headers.append(header)

    return science_frames, science_headers


def spread_1d_spectrum_to_2d(counts_s_pixel_convolved, channel_cfg, channel_cal, header=None):

    logging.info("Building single Science Frame: channel=%s.", channel_cfg.channel_name)

    nx = channel_cfg.x_pixels
    mode = channel_cfg.mode

    if len(counts_s_pixel_convolved) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s counts_len=%d nx=%d", channel_cfg.channel_name, int(len(counts_s_pixel_convolved)), int(nx))
        print(f"PROFILE SPREAD ERROR: channel={channel_cfg.channel_name} counts_len={len(counts_s_pixel_convolved)} nx={nx}")
        raise ValueError(f"Counts length {len(counts_s_pixel_convolved)} does not match nx {nx}")

    # no lookup or high resolution spectrograph spreading as of now.
    if mode == 1:

        if (
            channel_cal.spread_y_positions is not None
            and channel_cal.spread_y_weights is not None
            and channel_cal.spread_y_wavelengths is not None
        ):
            # wavelength dependent spreading will be used
            return _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel_cfg, channel_cal, header=header)

        else:
            # Gaussian dependent spreading
            return _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel_cfg, channel_cal, header=header)
           
    msg = f"mode={mode} not implemented yet (only mode=1 is supported)"
    logging.error(msg)
    raise NotImplementedError(msg)


def _spread_1d_to_2d_gaussian(counts_s_pixel_convolved, channel_cfg, channel_cal, header=None):

    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels
    spread_half_height = channel_cfg.spread_half_height_pix

    if spread_half_height <= 0:
        logging.error("SPREAD CONFIG ERROR: channel=%s no spread profile and spread_half_height_pix=%d", channel_cfg.channel_name, channel_cfg.spread_half_height_pix)
        raise ValueError("No cross-dispersion spreading configured")    

    # choose a fixed trace center (placeholder)
    y0 = ny // 2
    spatial_sigma_pix = float(channel_cfg.spread_half_height_pix)

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

    if header is not None:
        header.append(("MEAN",     float(image.mean()),                 "Mean value of the frame"))
        header.append(("MEDIAN",   float(np.median(image)),             "Median value of the frame"))
        header.append(("MAX",      float(image.max()),                  "Maximum value of the frame"))
        header.append(("MIN",      float(image.min()),                  "Minimum value of the frame"))
        header.append(("DARKVAL",  float(channel_cfg.dark_noise),       "Input dark value"))
        header.append(("B_OFFSET", float(channel_cfg.bias_offset),      "Bias offset used to generate frame"))
        header.append(("RNOISE",   float(channel_cfg.read_noise),       "Read noise sigma used to generate frame"))

    logging.info("GAUSSIAN SPREAD RESULT: channel=%s shape=(%d,%d) sum=%g", channel_cfg.channel_name, image.shape[0], image.shape[1], float(image.sum()))
    col_sums = image.sum(axis=0)
    logging.info("GAUSSIAN SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel_cfg.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))

    if not np.allclose(col_sums, counts_s_pixel_convolved, rtol=1e-10, atol=1e-12):
        logging.error("GAUSSIAN SPREAD CHECK FAILED: channel=%s column sums do not match input counts", channel_cfg.channel_name)
        raise ValueError("Gaussian spread column sum mismatch")

    return image, header


def _spread_1d_to_2d_profile(counts_s_pixel_convolved, channel_cfg, channel_cal, header=None):
    logging.info("WAVELENGTH DEPENDENT SPREAD: channel=%s spread_file=%s mode=1 profile detected but not yet implemented", channel_cfg.channel_name, channel_cfg.spread_profile_file)

    nx = channel_cfg.x_pixels
    ny = channel_cfg.y_pixels

    spread_y_pos = channel_cal.spread_y_positions
    spread_weigths = channel_cal.spread_y_weights
    spread_wavelengths = channel_cal.spread_y_wavelengths
    detector_wavelengths = channel_cal.wavelength

    # already in spread_1d_spectrum_to_2d
    # if spread_y_pos is None or spread_weigths is None or spread_wavelengths is None:
    #     logging.error("PROFILE SPREAD CONFIG ERROR: channel=%s missing spread arrays", channel_cfg.channel_name)
    #     raise ValueError("Missing spread profile arrays")

    if len(detector_wavelengths) != nx:
        logging.error("PROFILE SPREAD ERROR: channel=%s detector wavelength length mismatch", channel_cfg.channel_name)
        raise ValueError("Detector wavelength grid length mismatch")

    # already in effective_area_loader.py
    # if spread_weigths.ndim != 2:
    #     logging.error("PROFILE SPREAD CONFIG ERROR: channel=%s spread_y_weights must be 2D", channel_cfg.channel_name)
    #     raise ValueError("Invalid spread_y_weights dimension")

    # already in effective_area_loader.py
    # if spread_weigths.shape[0] != spread_y_pos.shape[0]:
    #     logging.error("PROFILE SPREAD CONFIG ERROR: channel=%s spread_y_positions and spread_y_weights row mismatch", channel_cfg.channel_name)
    #     raise ValueError("Spread profile row mismatch")

    # already in effective_area_loader.py
    # if spread_weigths.shape[1] != spread_wavelengths.shape[0]:
    #     logging.error("PROFILE SPREAD CONFIG ERROR: channel=%s spread_y_weights cols=%d != spread_y_wavelengths len=%d", channel_cfg.channel_name, int(spread_weigths.shape[1]), int(spread_wavelengths.shape[0]))
    #     raise ValueError("Spread profile column mismatch")

    dy = np.round(spread_y_pos).astype(np.int64)
    y0 = ny // 2

    logging.info("PROFILE SPREAD START: channel=%s spread_file=%s nx=%d ny=%d n_bins=%d y0=%d", channel_cfg.channel_name, channel_cfg.spread_profile_file, int(nx), int(ny), int(spread_wavelengths.shape[0]), int(y0))

    image = np.zeros((ny, nx), dtype=np.float64)

    for x in range(nx):
        lam = float(detector_wavelengths[x])
        j = int(np.argmin(np.abs(spread_wavelengths - lam)))
        lam_match = float(spread_wavelengths[j])
        delta = float(lam - lam_match)

        c = float(counts_s_pixel_convolved[x])
        for i in range(dy.shape[0]):
            y = int(y0 + dy[i])
            if 0 <= y < ny:
                image[y, x] += c * float(spread_weigths[i, j])
                # if x < 100:
                #     logging.info("PROFILE MATCH: channel=%s x=%d lam_det=%g lam_spread=%g delta=%g bin=%d counts_s_pixel_convolved[%d]=%g dy[%d]=%d y=%d weight=%g product=%g", channel_cfg.channel_name, int(x), lam, lam_match, delta, int(j), int(x), c, int(i), int(dy[i]), int(y0 + dy[i]), float(spread_weigths[i, j]), c * float(spread_weigths[i, j]))
    

    col_sums = image.sum(axis=0)
    logging.info("PROFILE SPREAD CHECK: channel=%s input_sum=%g image_sum=%g max_abs_diff=%g", channel_cfg.channel_name, float(np.sum(counts_s_pixel_convolved)), float(np.sum(image)), float(np.max(np.abs(col_sums - counts_s_pixel_convolved))))
    # if not np.allclose(col_sums, counts_s_pixel_convolved, rtol=1e-10, atol=1e-12):
    #     logging.error("PROFILE SPREAD CHECK FAILED: channel=%s column sums mismatch", channel_cfg.channel_name)
    #     raise ValueError("Profile spread column sum mismatch")

    if header is not None:
        header.append(("MEAN", float(image.mean()), "Mean value of the frame"))
        header.append(("MEDIAN", float(np.median(image)), "Median value of the frame"))
        header.append(("MAX", float(image.max()), "Maximum value of the frame"))
        header.append(("MIN", float(image.min()), "Minimum value of the frame"))


    logging.info("DEBUG 1D: min=%g max=%g s0=%g s1=%g s2=%g s3=%g s4=%g", float(np.min(counts_s_pixel_convolved)), float(np.max(counts_s_pixel_convolved)), float(counts_s_pixel_convolved[0]), float(counts_s_pixel_convolved[1]), float(counts_s_pixel_convolved[2]), float(counts_s_pixel_convolved[3]), float(counts_s_pixel_convolved[4]))
    logging.info("DEBUG y0: min=%g max=%g s0=%g s1=%g s2=%g s3=%g s4=%g", float(np.min(image[y0, :])), float(np.max(image[y0, :])), float(image[y0, 0]), float(image[y0, 1]), float(image[y0, 2]), float(image[y0, 3]), float(image[y0, 4]))
    logging.info("DEBUG colsum: min=%g max=%g s0=%g s1=%g s2=%g s3=%g s4=%g", float(np.min(image.sum(axis=0))), float(np.max(image.sum(axis=0))), float(image[:, 0].sum()), float(image[:, 1].sum()), float(image[:, 2].sum()), float(image[:, 3].sum()), float(image[:, 4].sum()))

    return image, header
    