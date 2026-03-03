import logging
import numpy as np
# import scipy.interpolate as si
from configs.channel_config import PhotometryChannel
from loaders.run_waltzer_context import RunContext
from utils.images import write_image_png

def spread_1d_photometry_to_2d(counts_s_px_nir: np.ndarray, channel: PhotometryChannel, ctx: RunContext) -> np.ndarray:
    # electrons_s_nir, npix = compute_photometry_signal_parameters(counts_s_px_convolved, channel)
    electrons_s_nir = float(np.sum(counts_s_px_nir))


    # 2) build a 2D stamp from the NIR spread file (fixed support from spread_positions)
    if channel.spread_positions is None:
        raise ValueError("NIR photometry: channel.spread_positions is None (spread file not loaded).")
    if channel.spread_y_weights is None:
        raise ValueError("NIR photometry: channel.spread_y_weights is None (spread file not loaded).")
    if channel.spread_x_weights is None:
        raise ValueError("NIR photometry: channel.spread_x_weights is None (spread file not loaded).")
    if channel.spread_anchors is None:
        raise ValueError("NIR photometry: channel.spread_anchors is None (spread file not loaded).")

    pos = channel.spread_positions
    spread_y_weights = channel.spread_y_weights
    spread_x_weights = channel.spread_x_weights
    anchors = channel.spread_anchors

    # map integer offsets -> row indices in the table
    pos_int = np.rint(pos).astype(int)
    idx_of = {int(v): i for i, v in enumerate(pos_int)}
    half = int(np.max(np.abs(pos_int)))  # should be 20 for your file
    size = 2 * half + 1                  # 41

    stamp = np.zeros((size, size), dtype=float)

    # fill stamp (dy, dx) in [-half..half]
    for iy, dy in enumerate(range(-half, half + 1)):
        row_y = idx_of.get(dy, None)
        if row_y is None:
            continue

        for ix, dx in enumerate(range(-half, half + 1)):
            row_x = idx_of.get(dx, None)
            if row_x is None:
                continue

            # r_arcsec = r_pix * float(channel.pixel_scale)

            # # interpolate over anchors at this radius, separately for x and y profiles
            # wy_val = float(np.interp(r_arcsec, anchors, spread_y_weights[row_y, :], left=spread_y_weights[row_y, 0], right=spread_y_weights[row_y, -1]))
            # wx_val = float(np.interp(r_arcsec, anchors, spread_x_weights[row_x, :], left=spread_x_weights[row_x, 0], right=spread_x_weights[row_x, -1]))

            # stamp[iy, ix] = max(0.0, wx_val) * max(0.0, wy_val)


            r_pix = float(np.sqrt(dx * dx + dy * dy))
            r_int = int(round(r_pix))
            if r_int > half:
                continue

            row_r = idx_of.get(r_int, None)
            if row_r is None:
                continue

            r_arcsec = r_pix * float(channel.pixel_scale)

            # use ONE value based on radius, not wx*wy
            val = float(np.interp(
                r_arcsec,
                anchors,
                spread_y_weights[row_r, :],
                left=spread_y_weights[row_r, 0],
                right=spread_y_weights[row_r, -1],
            ))

            stamp[iy, ix] = max(0.0, val)


    s = float(np.sum(stamp))
    if s <= 0.0:
        raise ValueError("NIR photometry: spread-based stamp sums to zero.")
    stamp /= s

    # 3) scale stamp to total rate and paste into detector frame
    psf_stamp_electrons_per_second = stamp * electrons_s_nir

    nir_image_e_s = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    
    if channel.source_position_x_arcsec != 0.0 or channel.source_position_y_arcsec != 0.0:
        raise NotImplementedError("Non-zero photometry source position not implemented yet.")
    
    cx_center = channel.x_pixels // 2
    cy_center = channel.y_pixels // 2

    dx_pix = float(channel.source_position_x_arcsec) / float(channel.pixel_scale)
    dy_pix = float(channel.source_position_y_arcsec) / float(channel.pixel_scale)

    cx = int(cx_center + dx_pix)
    cy = int(cy_center + dy_pix)

    paste_stamp_center(nir_image_e_s, psf_stamp_electrons_per_second, cx, cy)

    write_image_png(nir_image_e_s, "nir_image_frame_only", ctx, channel, show_stats=False)

    
    return nir_image_e_s

    # psf_stamp_norm = build_psf_stamp_from_radial_profile(channel.psf_radial_distance, channel.psf_radial_flux, npix)

    # psf_stamp_electrons_per_second = psf_stamp_norm * electrons_s_nir

    # nir_image_e_s = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    # if channel.source_position_x_arcsec != 0.0 or channel.source_position_y_arcsec != 0.0:
    #     raise NotImplementedError("Non-zero photometry source position not implemented yet.")
    # # center in pixels
    # cx_center = channel.x_pixels // 2
    # cy_center = channel.y_pixels // 2

    # # convert arcsec offsets to pixels
    # dx_pix = channel.source_position_x_arcsec / channel.pixel_scale
    # dy_pix = channel.source_position_y_arcsec / channel.pixel_scale

    # cx = int(cx_center + dx_pix)
    # cy = int(cy_center + dy_pix)
    # paste_stamp_center(nir_image_e_s, psf_stamp_electrons_per_second, cx, cy)

    # ctx.write_image_png.write_image(nir_image_e_s, "nir_image_frame_only", ctx, channel)

    # return nir_image_e_s



def compute_photometry_signal_parameters(counts_s_px: np.ndarray, channel:  PhotometryChannel,) -> tuple[float, float, float, float, int]:
    """
    From convolved counts per wavelength bin, compute:
        - band-integrated rate (e-/s)
        - total electrons for exposure
        - aperture radius (pix)
        - Gaussian sigma (pix)
        - PSF half-size npix
    """

    # Collapse to scalar band rate
    source_flux_s = float(np.sum(counts_s_px))

    # Aperture photometry
    radius = 0.5 * channel.aperture_pix
    # For a Gaussian profile, assume that the standard deviation is a
    # third of the aperture radius
    sigma = radius / 3.0

    # Map up to 5*sigma away from the center
    npix = int(5.0 * sigma)
    logging.info("Photometry: aperture_pix=%.3f radius=%.3f sigma=%.3f npix=%d stamp_size=%dx%d", channel.aperture_pix, radius, sigma, npix, 2*npix+1, 2*npix+1)
    return source_flux_s, npix


# def build_psf_stamp_from_radial_profile(psf_radial_distance: np.ndarray, psf_radial_flux: np.ndarray, npix: int) -> np.ndarray:
#     """
#     Returns a 2D PSF stamp of shape (2*npix+1, 2*npix+1), normalized so sum = 1.
#     Radial distance is in pixels.
#     """
#     if npix < 1:
#         raise ValueError(f"Invalid npix={npix}. Must be >= 1.")

#     logging.info("PSF stamp: building from radial profile (npix=%d, profile points=%d)", npix, len(psf_radial_distance))

#     psf_r = si.interp1d(psf_radial_distance, psf_radial_flux, kind="slinear", bounds_error=False, fill_value="extrapolate")

#     x = np.arange(-npix, npix + 1)
#     y = np.arange(-npix, npix + 1)
#     X, Y = np.meshgrid(x, y)

#     # r = np.sqrt(X**2 + Y**2)

#    # TBD: PSF center is hard-coded at (0,0)
#     x0 = y0 = 0.0
#     r2 = np.sqrt((X - x0)**2 + (Y - y0)**2)

#     psf = np.clip(psf_r(r2), 0.0, np.inf)

#     # psf = np.clip(psf_r(r), 0.0, np.inf)
#     s = float(np.sum(psf))
#     if s <= 0.0:
#         raise ValueError("PSF stamp sums to zero (check PSF profile values).")

#     psf /= s

#     logging.info("Photometry: PSF stamp shape=%s sum(before scale)=%g", psf.shape, np.sum(psf))

#     return psf

def paste_stamp_center(frame: np.ndarray, stamp: np.ndarray, cx: int, cy: int) -> None: 
    """
    In-place paste of stamp into frame, centered at (cx, cy).
    Crops automatically at edges.
    """
    h, w = frame.shape
    sh, sw = stamp.shape
    hy = sh // 2
    hx = sw // 2

    x0 = max(0, cx - hx)
    x1 = min(w, cx + hx + 1)
    y0 = max(0, cy - hy)
    y1 = min(h, cy + hy + 1)

    sx0 = x0 - (cx - hx)
    sx1 = sx0 + (x1 - x0)
    sy0 = y0 - (cy - hy)
    sy1 = sy0 + (y1 - y0)

    frame[y0:y1, x0:x1] += stamp[sy0:sy1, sx0:sx1]
    
    logging.info("Photometry: frame sum=%g max_pixel_rate=%g", np.sum(frame), np.max(frame))