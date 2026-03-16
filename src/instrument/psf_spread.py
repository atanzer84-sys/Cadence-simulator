import logging
import numpy as np
# import scipy.interpolate as si
from configs.channel_config import PhotometryChannel, Channel
from loaders.run_waltzer_context import RunContext
from utils.helpers import announce


def spread_1d_photometry_to_2d(counts_s_px_nir: np.ndarray, channel: PhotometryChannel, ctx: RunContext, announce_user: bool = True) -> np.ndarray:
    announce(f"Spreading 1D photometry counts to 2D detector image for channel {channel.channel_name}.", to_user=announce_user)

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

    psf_stamp = channel.psf_image * total_flux_electrons_per_second
    detector_image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    source_pixel_x, source_pixel_y = get_photometry_placement(channel)

    psf_center_x = int(channel.psf_center_x)
    psf_center_y = int(channel.psf_center_y)

    paste_psf_stamp( detector_image, psf_stamp, source_pixel_x, source_pixel_y, psf_center_x, psf_center_y)

    nonzero_y, nonzero_x = np.nonzero(detector_image)
    if nonzero_y.size > 0:
        logging.info("Channel %s: detector_image stats shape=%s nonzero=%d min=%e max=%e bbox_y=%d..%d bbox_x=%d..%d", channel.channel_name, detector_image.shape, int(nonzero_y.size), float(np.min(detector_image)), float(np.max(detector_image)), int(np.min(nonzero_y)), int(np.max(nonzero_y)), int(np.min(nonzero_x)), int(np.max(nonzero_x)))
    else:
        logging.info("Channel %s: detector_image stats shape=%s nonzero=0 min=%e max=%e bbox_y=none bbox_x=none", channel.channel_name, detector_image.shape, float(np.min(detector_image)), float(np.max(detector_image)))

    logging.info("Photometry PSF spread applied: channel=%s detector_shape=(%d,%d) source_pixel=(%d,%d summed NIR flux = %e electrons/s frame_sum=%e", channel.channel_name, channel.y_pixels, channel.x_pixels, source_pixel_x, source_pixel_y, total_flux_electrons_per_second, float(np.sum(detector_image)))
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

    frame[frame_y0:frame_y1, frame_x0:frame_x1] += psf_stamp[stamp_y0:stamp_y1, stamp_x0:stamp_x1]


def get_photometry_placement(channel: PhotometryChannel):
    detector_center_x = channel.x_pixels // 2
    detector_center_y = channel.y_pixels // 2

    source_offset_x_arcsec = float(channel.source_position_x_arcsec or 0.0)
    source_offset_y_arcsec = float(channel.source_position_y_arcsec or 0.0)

    pixel_scale_arcsec_per_pixel = float(channel.pixel_scale)

    source_offset_x_pixels = source_offset_x_arcsec / pixel_scale_arcsec_per_pixel
    source_offset_y_pixels = source_offset_y_arcsec / pixel_scale_arcsec_per_pixel

    source_pixel_x = int(round(detector_center_x + source_offset_x_pixels))
    source_pixel_y = int(round(detector_center_y + source_offset_y_pixels))

    return source_pixel_x, source_pixel_y


def compute_aperture_photometry(image: np.ndarray, channel: Channel):

    if not isinstance(channel, PhotometryChannel):
        return None

    # 1) determine the aperture radius from the PSF by finding the outermost pixel above the 1% intensity threshold and doubling that radius, then integrate the counts inside that circle centered at the target (Cc)
    # get the center of the target
    x0, y0 = get_photometry_placement(channel)
    # determine the circle radius by determining the 1% intensity threshold in the PSF
    psf = channel.psf_image
    center_y, center_x = channel.psf_center_y, channel.psf_center_x
    y_coord, x_coord = np.indices(psf.shape, dtype=np.int32)
    r = np.sqrt((y_coord - center_y)*(y_coord - center_y) + (x_coord - center_x)*(x_coord - center_x))
    threshold = 0.001 * psf.max()
    mask = psf >= threshold
    radius_psf_1_percent = r[mask].max()
    psf_radius = 2 * radius_psf_1_percent
    pixel_y, pixel_x = np.indices(image.shape, dtype=np.int32)
    dx = pixel_x - x0
    dy = pixel_y - y0
    # faster method than sqrt or **2
    distance_to_center_squared = dx*dx + dy*dy
    circle_mask = distance_to_center_squared <= psf_radius*psf_radius
    counts_circle = np.sum(image[circle_mask])

    # 2) consider an annulus, centered on the target, having the inner radius equal to the radius you used above and the outer radius equal to twice that of the inner radius
    radius_annulus_inner = psf_radius
    radius_annulus_outer = 2 * radius_annulus_inner

    # 3) integrate the counts inside the annulus (let's call this integral: Ca)
    inner_radius_squared = radius_annulus_inner * radius_annulus_inner
    outer_radius_squared = radius_annulus_outer * radius_annulus_outer

    outside_circle = distance_to_center_squared > inner_radius_squared
    inside_outer_radius = distance_to_center_squared <= outer_radius_squared

    annulus_mask = outside_circle & inside_outer_radius
    counts_annulus = np.sum(image[annulus_mask])

    # 4) count the number of pixels inside the inner circle (let's call this Nc)
    number_pixels_circle = np.sum(circle_mask)

    # 5) count the number of pixels inside the annulus (let's call this Na)
    number_pixels_annulus = np.sum(annulus_mask)

    # guard against division by zero
    if number_pixels_annulus == 0:
        return None

    # 6) rescale Ca so that it is for the same number of pixels as Cc, therefore compute: Ca_background=Ca*Nc/Na
    counts_background_circle = counts_annulus * number_pixels_circle / number_pixels_annulus

    # 7) finally, compute the background subtracted stellar counts as: C_star=Cc-Ca_background
    counts_star = counts_circle - counts_background_circle

    #8) white noise uncertainty attached to C_star: C_star_noise = sqrt( ( sqrt(Cc) )^2 + ( sqrt(Ca_background) )^2 ) = sqrt( Cc + Ca_background )
    counts_star_noise = np.sqrt(counts_circle + counts_background_circle)

    logging.info("Aperture photometry (%s): radius_psf_1_percent=%g psf_radius=%g radius_annulus_outer=%g Cc=%g Ca=%g Nc=%d Na=%d C_background=%g C_star=%g C_star_noise=%g", channel.channel_name, radius_psf_1_percent, psf_radius, radius_annulus_outer, counts_circle, counts_annulus, number_pixels_circle, number_pixels_annulus, counts_background_circle, counts_star, counts_star_noise)

    return counts_star, counts_star_noise, x0, y0, radius_annulus_inner, radius_annulus_outer