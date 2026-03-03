import logging
import numpy as np
import scipy.interpolate as si
from configs.channel_config import PhotometryChannel
from loaders.run_waltzer_context import RunContext

def photometry_rate_1d_to_2d(counts_s_px_convolved: np.ndarray, channel: PhotometryChannel, ctx: RunContext) -> np.ndarray:
    electrons_s_nir, npix = compute_photometry_signal_parameters(counts_s_px_convolved, channel)

    psf_stamp_norm = build_psf_stamp_from_radial_profile(channel.psf_radial_distance, channel.psf_radial_flux, npix)

    psf_stamp_electrons_per_second = psf_stamp_norm * electrons_s_nir

    nir_image_e_s = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    cx = channel.x_pixels // 2
    cy = channel.y_pixels // 2
    paste_stamp_center(nir_image_e_s, psf_stamp_electrons_per_second, cx, cy)

    ctx.write_image_png.write_image(nir_image_e_s, "nir_image_frame_only", ctx, channel)

    return nir_image_e_s



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


def build_psf_stamp_from_radial_profile(psf_radial_distance: np.ndarray, psf_radial_flux: np.ndarray, npix: int) -> np.ndarray:
    """
    Returns a 2D PSF stamp of shape (2*npix+1, 2*npix+1), normalized so sum = 1.
    Radial distance is in pixels.
    """
    if npix < 1:
        raise ValueError(f"Invalid npix={npix}. Must be >= 1.")

    logging.info("PSF stamp: building from radial profile (npix=%d, profile points=%d)", npix, len(psf_radial_distance))

    psf_r = si.interp1d(psf_radial_distance, psf_radial_flux, kind="slinear", bounds_error=False, fill_value="extrapolate")

    x = np.arange(-npix, npix + 1)
    y = np.arange(-npix, npix + 1)
    X, Y = np.meshgrid(x, y)

    # r = np.sqrt(X**2 + Y**2)

   # TBD: PSF center is hard-coded at (0,0)
    x0 = y0 = 0.0
    r2 = np.sqrt((X - x0)**2 + (Y - y0)**2)

    psf = np.clip(psf_r(r2), 0.0, np.inf)

    # psf = np.clip(psf_r(r), 0.0, np.inf)
    s = float(np.sum(psf))
    if s <= 0.0:
        raise ValueError("PSF stamp sums to zero (check PSF profile values).")

    psf /= s

    logging.info("Photometry: PSF stamp shape=%s sum(before scale)=%g", psf.shape, np.sum(psf))

    return psf

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