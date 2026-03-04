import logging
import numpy as np
# import scipy.interpolate as si
from configs.channel_config import PhotometryChannel
from loaders.run_waltzer_context import RunContext


def spread_1d_photometry_to_2d(counts_s_px_nir: np.ndarray, channel: PhotometryChannel, ctx: RunContext) -> np.ndarray:
    print(f"Spreading 1D photometry counts to 2D detector image for channel {channel.channel_name}.")
    logging.info("Spreading 1D photometry counts to 2D detector image for channel %s.", channel.channel_name)

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

    return detector_image

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

    frame_before_sum = float(np.sum(frame))
    stamp_sum = float(np.sum(psf_stamp))
    stamp_patch_sum = float(np.sum(psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]))

    logging.info("paste_psf_stamp frame_shape=%s stamp_shape=%s det=(%d,%d) psf_center=(%d,%d) stamp_left_top=(%d,%d) frame_xy=(x%d:%d,y%d:%d) stamp_xy=(x%d:%d,y%d:%d) stamp_sum=%0.18e stamp_patch_sum=%0.18e frame_sum_before=%0.18e", str(frame.shape), str(psf_stamp.shape), detector_center_x, detector_center_y, psf_center_x, psf_center_y, stamp_left, stamp_top, frame_x0, frame_x1, frame_y0, frame_y1, stamp_x0, stamp_x1, stamp_y0, stamp_y1, stamp_sum, stamp_patch_sum, frame_before_sum)

    frame[frame_y0:frame_y1, frame_x0:frame_x1] += psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]

    frame_after_sum = float(np.sum(frame))

    logging.info("paste_psf_stamp frame_sum_after=%0.18e delta=%0.18e", frame_after_sum, frame_after_sum - frame_before_sum)
