import logging
import numpy as np
# import scipy.interpolate as si
from configs.channel_config import PhotometryChannel
from loaders.run_waltzer_context import RunContext
from utils.images import write_image_png


def spread_1d_photometry_to_2d(counts_s_px_nir: np.ndarray, channel: PhotometryChannel, ctx: RunContext) -> np.ndarray:

    if channel.source_position_x_arcsec != 0.0 or channel.source_position_y_arcsec != 0.0:
        raise NotImplementedError(
            f"Channel {channel.channel_name}: off-axis photometry source not implemented "
            f"(x={channel.source_position_x_arcsec} arcsec, y={channel.source_position_y_arcsec} arcsec)"
        )
    if channel.psf_image is None:
        raise ValueError("PSF image not loaded")

    if channel.psf_center_x is None or channel.psf_center_y is None:
        raise ValueError("PSF center not loaded")

    total_flux_electrons_per_second = float(np.sum(counts_s_px_nir))
    logging.info("Channel %s: summed NIR flux = %e electrons/s", channel.channel_name, total_flux_electrons_per_second)

    psf_stamp = channel.psf_image * total_flux_electrons_per_second
    logging.info("Channel %s: psf_stamp shape=%s sum=%e", channel.channel_name, psf_stamp.shape, float(np.sum(psf_stamp)))
    np.savetxt(ctx.output_dir / f"{str(ctx.target_name).replace(' ', '_')}_{channel.channel_name}_psf_stamp.txt", psf_stamp, fmt="%.18e")
    write_image_png(psf_stamp, "psf_stamp", ctx, channel, show_stats=False)

    detector_image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)

    detector_center_x = channel.x_pixels // 2
    detector_center_y = channel.y_pixels // 2

    source_offset_x_arcsec = float(channel.source_position_x_arcsec or 0.0)
    source_offset_y_arcsec = float(channel.source_position_y_arcsec or 0.0)

    pixel_scale_arcsec_per_pixel = float(channel.pixel_scale)

    source_offset_x_pixels = source_offset_x_arcsec / pixel_scale_arcsec_per_pixel
    source_offset_y_pixels = source_offset_y_arcsec / pixel_scale_arcsec_per_pixel

    source_pixel_x = int(round(detector_center_x + source_offset_x_pixels))
    source_pixel_y = int(round(detector_center_y + source_offset_y_pixels))

    psf_center_x = int(channel.psf_center_x)
    psf_center_y = int(channel.psf_center_y)

    paste_psf_stamp( detector_image, psf_stamp, source_pixel_x, source_pixel_y, psf_center_x, psf_center_y)

    logging.info("Channel %s: PSF placed at detector pixel (%d,%d) flux=%e frame_sum=%e", channel.channel_name, source_pixel_x, source_pixel_y, total_flux_electrons_per_second, float(np.sum(detector_image)))
    nonzero_y, nonzero_x = np.nonzero(detector_image)
    if nonzero_y.size > 0:
        logging.info("Channel %s: detector_image stats shape=%s nonzero=%d min=%e max=%e bbox_y=%d..%d bbox_x=%d..%d", channel.channel_name, detector_image.shape, int(nonzero_y.size), float(np.min(detector_image)), float(np.max(detector_image)), int(np.min(nonzero_y)), int(np.max(nonzero_y)), int(np.min(nonzero_x)), int(np.max(nonzero_x)))
    else:
        logging.info("Channel %s: detector_image stats shape=%s nonzero=0 min=%e max=%e bbox_y=none bbox_x=none", channel.channel_name, detector_image.shape, float(np.min(detector_image)), float(np.max(detector_image)))

    np.savetxt(ctx.output_dir / f"{str(ctx.target_name).replace(' ', '_')}_{channel.channel_name}_detector_image.txt", detector_image, fmt="%.18e")
    write_image_png(detector_image, "detector_image", ctx, channel, show_stats=False, use_percentile_scaling=False)

    return detector_image

# def spread_1d_photometry_to_2d(counts_s_px_nir: np.ndarray, channel: PhotometryChannel, ctx: RunContext) -> np.ndarray:

#     if channel.psf_image is None:
#         raise ValueError(f"Channel {channel.channel_name}: psf_image is not loaded")

#     # total electrons/s for the whole band
#     total_e_s = float(np.sum(counts_s_px_nir))

#     # scaled PSF stamp (psf_image is normalized so sum == 1)
#     stamp = channel.psf_image * total_e_s

#     # detector image
#     out = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)

#     # detector center
#     cx = int(channel.x_pixels // 2)
#     cy = int(channel.y_pixels // 2)

#     # paste stamp at detector center
#     paste_stamp_center(out, stamp, cx, cy)

#     logging.info("Channel %s: photometry spread applied cx=%d cy=%d total_e_s=%e stamp_sum=%e frame_sum=%e", channel.channel_name, cx, cy, total_e_s, float(np.sum(stamp)), float(np.sum(out)))

#     write_image_png(out, "nir_image_frame_only", ctx, channel, show_stats=False)
#     logging.info("total_e_s=%e peak_psf=%e peak_stamp=%e", total_e_s, np.max(channel.psf_image), np.max(stamp))
#     return out

    # electrons_s_nir = float(np.sum(counts_s_px_nir))


    # # 2) build a 2D stamp from the NIR spread file (fixed support from spread_positions)
    # if channel.spread_positions is None:
    #     raise ValueError("NIR photometry: channel.spread_positions is None (spread file not loaded).")
    # if channel.spread_y_weights is None:
    #     raise ValueError("NIR photometry: channel.spread_y_weights is None (spread file not loaded).")
    # if channel.spread_x_weights is None:
    #     raise ValueError("NIR photometry: channel.spread_x_weights is None (spread file not loaded).")
    # if channel.spread_anchors is None:
    #     raise ValueError("NIR photometry: channel.spread_anchors is None (spread file not loaded).")

    # pos = channel.spread_positions
    # spread_y_weights = channel.spread_y_weights
    # spread_x_weights = channel.spread_x_weights
    # anchors = channel.spread_anchors

    # # map integer offsets -> row indices in the table
    # pos_int = np.rint(pos).astype(int)
    # idx_of = {int(v): i for i, v in enumerate(pos_int)}
    # half = int(np.max(np.abs(pos_int)))
    # size = 2 * half + 1 # 41

    # stamp = np.zeros((size, size), dtype=float)

    # # fill stamp (dy, dx) in [-half..half]
    # for iy, dy in enumerate(range(-half, half + 1)):
    #     row_y = idx_of.get(dy, None)
    #     if row_y is None:
    #         continue

    #     for ix, dx in enumerate(range(-half, half + 1)):
    #         row_x = idx_of.get(dx, None)
    #         if row_x is None:
    #             continue

    #         # # radius from center in pixels, then in arcsec (anchors are in arcsec)
    #         # r_pix = float(np.sqrt(dx * dx + dy * dy))
    #         # r_arcsec = r_pix * float(channel.pixel_scale)

    #         # # interpolate over anchors at this radius, separately for x and y profiles
    #         # wy_val = float(np.interp(r_arcsec, anchors, spread_y_weights[row_y, :], left=spread_y_weights[row_y, 0], right=spread_y_weights[row_y, -1]))
    #         # wx_val = float(np.interp(r_arcsec, anchors, spread_x_weights[row_x, :], left=spread_x_weights[row_x, 0], right=spread_x_weights[row_x, -1]))

    #         # stamp[iy, ix] = max(0.0, wx_val) * max(0.0, wy_val)


    #         r_pix = float(np.sqrt(dx * dx + dy * dy))
    #         r_int = int(round(r_pix))
    #         if r_int > half:
    #             continue

    #         row_r = idx_of.get(r_int, None)
    #         if row_r is None:
    #             continue

    #         r_arcsec = r_pix * float(channel.pixel_scale)

    #         # use ONE value based on radius, not wx*wy
    #         val = float(np.interp(
    #             r_arcsec,
    #             anchors,
    #             spread_y_weights[row_r, :],
    #             left=spread_y_weights[row_r, 0],
    #             right=spread_y_weights[row_r, -1],
    #         ))

    #         stamp[iy, ix] = max(0.0, val)
    # s = float(np.sum(stamp))
    # if s <= 0.0:
    #     raise ValueError("NIR photometry: spread-based stamp sums to zero.")
    # stamp /= s

    # # 3) scale stamp to total rate and paste into detector frame
    # psf_stamp_electrons_per_second = stamp * electrons_s_nir

    # nir_image_e_s = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    
    # if channel.source_position_x_arcsec != 0.0 or channel.source_position_y_arcsec != 0.0:
    #     raise NotImplementedError("Non-zero photometry source position not implemented yet.")
    
    # cx_center = channel.x_pixels // 2
    # cy_center = channel.y_pixels // 2

    # dx_pix = float(channel.source_position_x_arcsec) / float(channel.pixel_scale)
    # dy_pix = float(channel.source_position_y_arcsec) / float(channel.pixel_scale)

    # cx = int(cx_center + dx_pix)
    # cy = int(cy_center + dy_pix)

    # paste_stamp_center(nir_image_e_s, psf_stamp_electrons_per_second, cx, cy)

    # write_image_png(nir_image_e_s, "nir_image_frame_only", ctx, channel, show_stats=False)

    
    # return nir_image_e_s

    # # psf_stamp_norm = build_psf_stamp_from_radial_profile(channel.psf_radial_distance, channel.psf_radial_flux, npix)

    # # psf_stamp_electrons_per_second = psf_stamp_norm * electrons_s_nir

    # # nir_image_e_s = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    # # if channel.source_position_x_arcsec != 0.0 or channel.source_position_y_arcsec != 0.0:
    # #     raise NotImplementedError("Non-zero photometry source position not implemented yet.")
    # # # center in pixels
    # # cx_center = channel.x_pixels // 2
    # # cy_center = channel.y_pixels // 2

    # # # convert arcsec offsets to pixels
    # # dx_pix = channel.source_position_x_arcsec / channel.pixel_scale
    # # dy_pix = channel.source_position_y_arcsec / channel.pixel_scale

    # # cx = int(cx_center + dx_pix)
    # # cy = int(cy_center + dy_pix)
    # # paste_stamp_center(nir_image_e_s, psf_stamp_electrons_per_second, cx, cy)

    # # ctx.write_image_png.write_image(nir_image_e_s, "nir_image_frame_only", ctx, channel)

    # # return nir_image_e_s



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

# def paste_stamp_center(frame: np.ndarray, stamp: np.ndarray, cx: int, cy: int) -> None: 
#     """
#     In-place paste of stamp into frame, centered at (cx, cy).
#     Crops automatically at edges.
#     """
#     h, w = frame.shape
#     sh, sw = stamp.shape
#     hy = sh // 2
#     hx = sw // 2

#     x0 = max(0, cx - hx)
#     x1 = min(w, cx + hx + 1)
#     y0 = max(0, cy - hy)
#     y1 = min(h, cy + hy + 1)

#     sx0 = x0 - (cx - hx)
#     sx1 = sx0 + (x1 - x0)
#     sy0 = y0 - (cy - hy)
#     sy1 = sy0 + (y1 - y0)

#     frame[y0:y1, x0:x1] += stamp[sy0:sy1, sx0:sx1]
    
#     logging.info("Photometry: frame sum=%g max_pixel_rate=%g", np.sum(frame), np.max(frame))

def paste_stamp_center(frame: np.ndarray, stamp: np.ndarray, cx: int, cy: int) -> None:
    """
    In-place paste of stamp into frame, centered at (cx, cy).
    Works for odd and even stamp sizes.
    Crops automatically at edges.
    """
    h, w = frame.shape
    sh, sw = stamp.shape

    # Define top-left so that the stamp is centered on (cx, cy)
    # For even sizes, this chooses the "left/up" center convention.
    x0 = cx - (sw // 2)
    y0 = cy - (sh // 2)
    x1 = x0 + sw
    y1 = y0 + sh

    # Crop to frame bounds
    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(w, x1)
    fy1 = min(h, y1)

    # Corresponding crop in stamp coordinates
    sx0 = fx0 - x0
    sy0 = fy0 - y0
    sx1 = sx0 + (fx1 - fx0)
    sy1 = sy0 + (fy1 - fy0)

    if fx0 >= fx1 or fy0 >= fy1:
        logging.info("Photometry: stamp fully outside frame (cx=%d cy=%d)", cx, cy)
        return

    frame[fy0:fy1, fx0:fx1] += stamp[sy0:sy1, sx0:sx1]

    logging.info("Photometry: frame sum=%g max_pixel_rate=%g", np.sum(frame), np.max(frame))

def paste_psf_stamp(frame: np.ndarray, psf_stamp: np.ndarray, detector_center_x: int, detector_center_y: int, psf_center_x: int, psf_center_y: int) -> None:

    stamp_height, stamp_width = psf_stamp.shape
    frame_height, frame_width = frame.shape

    stamp_left = detector_center_x - psf_center_x
    stamp_top = detector_center_y - psf_center_y

    stamp_right = stamp_left + stamp_width
    stamp_bottom = stamp_top + stamp_height

    frame_x0 = max(0, stamp_left)
    frame_y0 = max(0, stamp_top)
    frame_x1 = min(frame_width, stamp_right)
    frame_y1 = min(frame_height, stamp_bottom)

    stamp_x0 = frame_x0 - stamp_left
    stamp_y0 = frame_y0 - stamp_top
    stamp_x1 = stamp_x0 + (frame_x1 - frame_x0)
    stamp_y1 = stamp_y0 + (frame_y1 - frame_y0)

    if frame_x0 >= frame_x1 or frame_y0 >= frame_y1:
        logging.info("paste_psf_stamp NO_OVERLAP frame_shape=%s stamp_shape=%s det=(%d,%d) psf_center=(%d,%d) stamp_left_top=(%d,%d)", str(frame.shape), str(psf_stamp.shape), detector_center_x, detector_center_y, psf_center_x, psf_center_y, stamp_left, stamp_top)

        return

    # frame[frame_y0:frame_y1, frame_x0:frame_x1] += \
        # psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]

    frame_before_sum = float(np.sum(frame))
    stamp_sum = float(np.sum(psf_stamp))
    stamp_patch_sum = float(np.sum(psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]))

    logging.info("paste_psf_stamp frame_shape=%s stamp_shape=%s det=(%d,%d) psf_center=(%d,%d) stamp_left_top=(%d,%d) frame_xy=(x%d:%d,y%d:%d) stamp_xy=(x%d:%d,y%d:%d) stamp_sum=%0.18e stamp_patch_sum=%0.18e frame_sum_before=%0.18e", str(frame.shape), str(psf_stamp.shape), detector_center_x, detector_center_y, psf_center_x, psf_center_y, stamp_left, stamp_top, frame_x0, frame_x1, frame_y0, frame_y1, stamp_x0, stamp_x1, stamp_y0, stamp_y1, stamp_sum, stamp_patch_sum, frame_before_sum)

    frame[frame_y0:frame_y1, frame_x0:frame_x1] += psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]

    frame_after_sum = float(np.sum(frame))

    logging.info("paste_psf_stamp frame_sum_after=%0.18e delta=%0.18e", frame_after_sum, frame_after_sum - frame_before_sum)
