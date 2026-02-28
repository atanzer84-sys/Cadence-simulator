import logging
import numpy as np
from configs.global_config import GlobalConfig
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
from instrument.prepare_detector_images import convert_flux_to_photons
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel
from utils.constants import ARCSEC2_PER_SR
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

        np.savetxt(
            ctx.output_dir / "PY_background_raw.txt",
            np.column_stack((wl_bg, flux_bg)),
        )
        # IDL:
        # background_in = data_bg[1,*] * 25 * 5.03e7 * data_bg[0,*]
        # from ergs/s/cm2/A to photons/s/cm2/A

        # Convert surface brightness to “per sky pixel” by multiplying by sky pixel area:
        flux_bg_per_sky_pixel = flux_bg * channel.sky_pixel_area_arcsec2

        # convert photons per Angstrom into photons per pixel - converting to photons/s/cm2/per pixel
        photons_per_A_per_sky_pixel = convert_flux_to_photons(flux_bg_per_sky_pixel, wl_bg)
        np.savetxt(
            ctx.output_dir / f"PY_1_background_photons_{channel.channel_name}.txt",
            np.column_stack((wl_bg, photons_per_A_per_sky_pixel)),
        )


        # 3) IDL interpol(...) equivalent onto the grid IDL uses for the image row
        # Use the same target wavelength grid that your IDL variable `wavelength` represents.
        # In your Python context, that is almost certainly:
        # spline = CubicSpline(wl_bg, photons_per_A_per_sky_pixel, extrapolate=True)  # or extrapolate=False if you want to forbid it
        # background = spline(channel.effective_area_wavelength)

        # background = np.interp(channel.effective_area_wavelength, wl_bg, photons_per_A_per_sky_pixel)

        spline = CubicSpline(
            wl_bg,
            photons_per_A_per_sky_pixel,
            bc_type="natural",      # matches IDL much better
            extrapolate=True
        )

        background = spline(channel.effective_area_wavelength)
        np.savetxt(
            ctx.output_dir / f"PY_2_background_interpolate_{channel.channel_name}.txt",
            np.column_stack((channel.effective_area_wavelength, background)),
        )

        # aeff2 = channel.effective_area
        aeff_spline = CubicSpline(
            channel.effective_area_wavelength,
            channel.effective_area,
            bc_type="natural",
            extrapolate=True
        )

        aeff2 = aeff_spline(channel.effective_area_wavelength)


        np.savetxt(
            ctx.output_dir / f"PY_3_aeff2_{channel.channel_name}.txt",
            np.column_stack((channel.effective_area_wavelength, aeff2)),
            fmt="%.16e",
        )

        scaled_spectrum = background * aeff2

        np.savetxt(
            ctx.output_dir / f"PY_4_scaled_spectrum_{channel.channel_name}.txt",
            np.column_stack((channel.effective_area_wavelength, scaled_spectrum)),
            fmt="%.16e",
        )
        # counts_s_px_convolved = counts_per_s_px_conv_per_channel(photons_per_A_per_sky_pixel, wl_bg, channel, star, ctx, filename_suffix= "Background")

        # np.savetxt(
        #     ctx.output_dir / "PY_background_counts.txt",
        #     np.column_stack((channel.effective_area_wavelength, counts_s_px_convolved)),
        # )
        image[:, :] = scaled_spectrum[np.newaxis, :]
        image*=channel.exposure_s

        logging.debug("Default background 2D image shape: %s", image.shape)
        ctx.write_image_png.write_image(image, "BACKGROUND_only", ctx, channel)


    if channel.background_type == "calc":


        # sunpos, jd, ra_s, dec_s
        # t = Time(jd, format="jd", scale="utc")
        timestamp = ctx.timestamp
        time = Time(timestamp, scale='utc')
        sun_icrs = get_sun(time).icrs
        ra_s = float(sun_icrs.ra.to_value(u.deg))
        dec_s = float(sun_icrs.dec.to_value(u.deg))
        ra = star.right_ascension
        dec = star.declination
        
        # euler, ra_s, dec_s, elb_s, ela_s, select=3   (ICRS -> ecliptic lon/lat)
        sun_ecl = SkyCoord(ra=ra_s * u.deg, dec=dec_s * u.deg, frame="icrs").transform_to(BarycentricTrueEcliptic())
        elb_s = float(sun_ecl.lon.to_value(u.deg))
        ela_s = float(sun_ecl.lat.to_value(u.deg))

        # length=file_lines(file_zod)
        # data_zod=dblarr(12,length)
        # openr/readf/close
        data_zod = channel.zod_dist

        # euler, ra, dec, elb, ela, select=3   (target ICRS -> ecliptic lon/lat)
        targ_ecl = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(BarycentricTrueEcliptic())
        elb = float(targ_ecl.lon.to_value(u.deg))
        ela = float(targ_ecl.lat.to_value(u.deg))


        # elb_h = fix(elb - elb_s)
        elb_h = int(elb - elb_s)

        if elb_h > 180:
            elb_h = 360 - elb_h

        # ela_h = fix(ela)
        ela_h = int(ela)

        # i = 0
        i = 0
        while data_zod[i, 0] < ela_h:
            i += 1
            if i >= data_zod.shape[0] - 1:
                break

        if i > 0:
            if (data_zod[i, 0] - ela_h) > (ela_h - data_zod[i - 1, 0]):
                i -= 1

        j = 0
        while data_zod[0, j] < elb_h:
            j += 1
            if j >= data_zod.shape[1] - 1:
                break

        if j > 0:
            if (data_zod[0, j] - elb_h) > (elb_h - data_zod[0, j - 1]):
                j -= 1

        zod_value = data_zod[i, j]

        wl_sol = channel.zod_spectrum_wavelength
        flux_sol = channel.zod_spectrum_flux

        zod_spectrum = zod_value * flux_sol

        spline = CubicSpline(wl_sol, zod_spectrum, extrapolate=True)
        background = spline(channel.effective_area_wavelength)

        background*= channel.sky_pixel_area_arcsec2 / ARCSEC2_PER_SR 
        background*= channel.effective_area * channel.pixel_scale


        image[:, :] = background[np.newaxis, :]

        logging.debug("Default background 2D image shape: %s", image.shape)
        ctx.write_image_png.write_image(image, "BACKGROUND_only", ctx, channel)


    return image
