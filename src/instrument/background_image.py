import logging
import numpy as np
from configs.global_config import GlobalConfig
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
from instrument.prepare_detector_images import convert_flux_to_photons
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel

from astropy.time import Time
from astropy.coordinates import SkyCoord, get_sun, BarycentricTrueEcliptic
import astropy.units as u
from scipy.interpolate import CubicSpline


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

        logging.debug("Default background 2D image shape: %s", image.shape)
        ctx.write_image_png.write_image(image, "BACKGROUND_only", ctx, channel)


    # if channel.background_type == "calc":

    #     # -----------------------------
    #     # Sun position at jd
    #     # -----------------------------

    #     timestamp = ctx.timestamp
    #     time = Time(t, scale='utc')
    #     jd = time.jd
    #     ra = star.ra_deg
    #     dec = star.dec_deg
    #     sun = get_sun(time).icrs

    #     ra_s = float(sun.ra.to_value(u.deg))
    #     dec_s = float(sun.dec.to_value(u.deg))

    #     # Convert Sun to ecliptic
    #     sun_coord = SkyCoord(ra=ra_s * u.deg, dec=dec_s * u.deg, frame="icrs")
    #     sun_ecl = sun_coord.transform_to(BarycentricTrueEcliptic())
    #     elb_s = float(sun_ecl.lon.to_value(u.deg))
    #     ela_s = float(sun_ecl.lat.to_value(u.deg))

    #     # Convert target to ecliptic
    #     targ_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    #     targ_ecl = targ_coord.transform_to(BarycentricTrueEcliptic())
    #     elb = float(targ_ecl.lon.to_value(u.deg))
    #     ela = float(targ_ecl.lat.to_value(u.deg))

    #     # -----------------------------
    #     # Zodiacal geometry
    #     # -----------------------------
    #     elb_h = int(elb - elb_s)
    #     if elb_h > 180:
    #         elb_h = 360 - elb_h

    #     ela_h = int(ela)

    #     # -----------------------------
    #     # Read zodiacal lookup table
    #     # -----------------------------
    #     file_zod = channel.background_zodiacal_file
    #     data_zod = np.loadtxt(file_zod)

    #     # IDL nearest index logic
    #     i = 0
    #     while i < data_zod.shape[0] - 1 and data_zod[i, 0] < ela_h:
    #         i += 1

    #     if i > 0:
    #         if (data_zod[i, 0] - ela_h) > (ela_h - data_zod[i - 1, 0]):
    #             i -= 1

    #     j = 0
    #     while j < data_zod.shape[1] - 1 and data_zod[0, j] < elb_h:
    #         j += 1

    #     if j > 0:
    #         if (data_zod[0, j] - elb_h) > (elb_h - data_zod[0, j - 1]):
    #             j -= 1

    #     zod_value = data_zod[i, j]

    #     # -----------------------------
    #     # Solar reference spectrum
    #     # -----------------------------
    #     file_zods = channel.background_solar_file
    #     data_zods = np.loadtxt(file_zods)

    #     wl_sol = data_zods[:, 0]
    #     flux_sol = data_zods[:, 1]

    #     zod_spectrum = zod_value * flux_sol

    #     # -----------------------------
    #     # Interpolation (/SPLINE)
    #     # -----------------------------
    #     wavelength = channel.effective_area_wavelength
    #     spline = CubicSpline(wl_sol, zod_spectrum, extrapolate=True)
    #     background = spline(wavelength)

    #     # -----------------------------
    #     # Convert to photons/s/cm2/pixel
    #     # -----------------------------
    #     background = background * 25.0 / 4.25e10

    #     # -----------------------------
    #     # Propagate through detector path
    #     # -----------------------------
    #     counts_s_px_convolved = counts_per_s_px_conv_per_channel(
    #         background,
    #         wavelength,
    #         channel,
    #         star,
    #         ctx,
    #         filename_suffix="Background"
    #     )

    #     image[:, :] = counts_s_px_convolved[np.newaxis, :]
    #     image *= channel.exposure_s
    return image
