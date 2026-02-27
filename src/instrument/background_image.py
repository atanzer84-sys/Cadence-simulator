import logging
import numpy as np
from configs.global_config import GlobalConfig
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
from instrument.prepare_detector_images import convert_flux_to_photons
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel

# try:
#     from astropy.time import Time
#     from astropy.coordinates import SkyCoord, get_sun, BarycentricTrueEcliptic
#     import astropy.units as u
#     _HAVE_ASTROPY = True
# except Exception:
#     _HAVE_ASTROPY = False

# # Optional: for spline interpolation (matches IDL /SPLINE best)
# try:
#     from scipy.interpolate import CubicSpline
#     _HAVE_SCIPY = True
# except Exception:
#     _HAVE_SCIPY = False


# def _interp_like_idl_spline(y: np.ndarray, x: np.ndarray, x_new: np.ndarray) -> np.ndarray:
#     """
#     IDL interpol(...,/SPLINE) equivalent-ish.
#     Uses CubicSpline if available, else linear np.interp.
#     """
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     x_new = np.asarray(x_new, dtype=float)

#     # Ensure strictly increasing x for spline
#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]

#     if _HAVE_SCIPY and x.shape[0] >= 4:
#         cs = CubicSpline(x, y, extrapolate=True)
#         return cs(x_new)

#     # Fallback: linear interpolation
#     return np.interp(x_new, x, y, left=y[0], right=y[-1])


# def _radec_to_ecliptic_lonlat_deg(ra_deg: float, dec_deg: float) -> tuple[float, float]:
#     if not _HAVE_ASTROPY:
#         raise RuntimeError("background_type='calc' needs astropy installed (Sun position + ecliptic conversion).")
#     c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
#     ecl = c.transform_to(BarycentricTrueEcliptic())
#     lon = float(ecl.lon.to_value(u.deg))
#     lat = float(ecl.lat.to_value(u.deg))
#     return lon, lat


# def _sun_radec_deg_from_jd(jd: float) -> tuple[float, float]:
#     if not _HAVE_ASTROPY:
#         raise RuntimeError("background_type='calc' needs astropy installed (Sun position + ecliptic conversion).")
#     t = Time(jd, format="jd", scale="utc")
#     sun = get_sun(t).icrs
#     return float(sun.ra.to_value(u.deg)), float(sun.dec.to_value(u.deg))


# def _nearest_index_1d(sorted_vals: np.ndarray, target: float) -> int:
#     """
#     IDL code does a bracket search + nearest neighbor.
#     This does nearest neighbor directly (same intent).
#     """
#     sorted_vals = np.asarray(sorted_vals, dtype=float)
#     return int(np.argmin(np.abs(sorted_vals - target)))


def generate_Background_Image(channel: SpectroscopyChannel, ctx: RunContext, cfg: GlobalConfig, star: Star ) -> np.ndarray:

    nx = channel.x_pixels
    ny = channel.y_pixels

    logging.info("Generating background image for channel %s with size %dx%d (nx x ny).", channel.channel_name, nx, ny)

    image = np.zeros((ny, nx), dtype=float)

    if channel.background_type is None:
        logging.info("Background disabled: 'background_type' is None. Returning zero background image.")
        return image

    if channel.background_type == "default":

        wl_bg = channel.background_wavelength # Å
        flux_bg = channel.background_flux # erg / s / cm² / Å
        logging.info("Background type 'default': using background spectrum with %d wavelength points and %d flux points.", wl_bg.size, flux_bg.size)

        wl_bg = np.asarray(wl_bg)
        flux_bg = np.asarray(flux_bg)

        # IDL:
        # background_in = data_bg[1,*] * 25 * 5.03e7 * data_bg[0,*]
        # from ergs/s/cm2/A to photons/s/cm2/A
        # Convert surface brightness to “per sky pixel” by multiplying by sky pixel area:
        flux_bg_per_sky_pixel = flux_bg * channel.sky_pixel_area_arcsec2

        # convert photons per Angstrom into photons per pixel - converting to photons/s/cm2/per pixel
        photons_per_A_per_sky_pixel = convert_flux_to_photons(flux_bg_per_sky_pixel, wl_bg)

        counts_s_px_convolved = counts_per_s_px_conv_per_channel(photons_per_A_per_sky_pixel, wl_bg, channel, star, ctx, filename_suffix= "Background")

        image[:, :] = counts_s_px_convolved[np.newaxis, :]
        image *= channel.exposure_s

        logging.debug("Default background 2D image shape: %s", image.shape)
        ctx.write_image_png.write_image(image, "BACKGROUND_only", ctx, channel)

    return image
