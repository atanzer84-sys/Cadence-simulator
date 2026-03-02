import logging
import numpy as np
import scipy.interpolate as si

def build_psf_stamp_from_radial_profile(psf_radial_distance: np.ndarray, psf_radial_flux: np.ndarray, npix: int) -> np.ndarray:
    """
    Returns a 2D PSF stamp of shape (2*npix+1, 2*npix+1), normalized so sum = 1.
    Radial distance is in pixels.
    """
    if psf_radial_distance is None or psf_radial_flux is None:
        raise ValueError("PSF radial profile not loaded (psf_radial_distance/psf_radial_flux).")

    if npix < 1:
        raise ValueError(f"Invalid npix={npix}. Must be >= 1.")


    logging.info(
        "PSF stamp: building from radial profile (npix=%d, profile points=%d)",
        npix,
        len(psf_radial_distance),
    )
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
    side = 2 * npix + 1
    logging.info("PSF stamp: built shape (%d, %d), normalized sum=1.0", side, side)
    return psf

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