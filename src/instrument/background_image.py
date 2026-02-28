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


def generate_Background_Image(channel: SpectroscopyChannel, ctx: RunContext, star: Star ) -> np.ndarray:

    nx = channel.x_pixels
    ny = channel.y_pixels

    logging.info("Generating background image for channel %s with size %dx%d (nx x ny).", channel.channel_name, nx, ny)

    image = np.zeros((ny, nx), dtype=float)

    if channel.background_type is None:
        logging.info("Background disabled: 'background_type' is None. Returning zero background image.")
        return image

    if channel.background_type == "default":
       background = generate_background_default_image(channel)

    if channel.background_type == "calc":
       background = generate_background_calculated_image(channel, star)

    background = background * channel.effective_area
    image[:, :] = background[np.newaxis, :]
    image*=channel.exposure_s

    logging.debug("Default background 2D image shape: %s", image.shape)
    ctx.write_image_png.write_image(image, "BACKGROUND_only", ctx, channel)

    return image


def generate_background_default_image(channel: SpectroscopyChannel):
    wl_bg = channel.background_wavelength # Å
    flux_bg = channel.background_flux # erg / s / cm² / Å
    logging.info("Background type 'default': using background spectrum with %d wavelength points and %d flux points.", wl_bg.size, flux_bg.size)

    # IDL:
    # background_in = data_bg[1,*] * 25 * 5.03e7 * data_bg[0,*]
    # from ergs/s/cm2/A to photons/s/cm2/A
    # Convert surface brightness to “per sky pixel” by multiplying by sky pixel area:
    flux_bg_per_sky_pixel = flux_bg * channel.sky_pixel_area_arcsec2

    # convert photons per Angstrom into photons per pixel - converting to photons/s/cm2/per pixel
    photons_per_A_per_sky_pixel = convert_flux_to_photons(flux_bg_per_sky_pixel, wl_bg)

    spline = CubicSpline(wl_bg, photons_per_A_per_sky_pixel, bc_type="natural", extrapolate=True)
    background = spline(channel.effective_area_wavelength)

    return background


def generate_background_calculated_image(channel: SpectroscopyChannel, star: Star):

    # sunpos, jd, ra_s, dec_s
    # timestamp = ctx.timestamp
    # time = Time(timestamp, scale='utc')
    # jd = float(time.jd)
    jd = 2457095.5
    time = Time(jd, format="jd", scale="utc")
    sun_icrs = get_sun(time).icrs
    ra_s = float(sun_icrs.ra.to_value(u.deg))
    dec_s = float(sun_icrs.dec.to_value(u.deg))
    ra = star.right_ascension
    dec = star.declination

    # euler, ra_s, dec_s, elb_s, ela_s, select=3   (ICRS -> ecliptic lon/lat)
    sun_ecl = SkyCoord(ra=ra_s * u.deg, dec=dec_s * u.deg, frame="icrs").transform_to(BarycentricTrueEcliptic())
    elb_s = float(sun_ecl.lon.to_value(u.deg))
    ela_s = float(sun_ecl.lat.to_value(u.deg))

    data_zod = np.asarray(channel.zod_dist, dtype=np.float64)

    # euler, ra, dec, elb, ela, select=3   (target ICRS -> ecliptic lon/lat)
    targ_ecl = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(BarycentricTrueEcliptic())
    elb = float(targ_ecl.lon.to_value(u.deg))
    ela = float(targ_ecl.lat.to_value(u.deg))

    elb_h = int(elb - elb_s)
    if elb_h > 180:
        elb_h = 360 - elb_h
    ela_h = int(ela)

    def _fmt(x: float) -> str:
        return f"{x:.16e}".strip()

    logging.info("Background type 'calc': running calculated (zodiacal) background image."
        "targ_ra=%s targ_dec=%s targ_elb=%s targ_ela=%s sunpos_jd=%s sunpos_ra=%s sunpos_dec=%s sunpos_elb=%s sunpos_ela=%s elb_h=%s ela_h=%s", _fmt(ra), _fmt(dec), _fmt(elb), _fmt(ela), _fmt(jd), _fmt(ra_s), _fmt(dec_s), _fmt(elb_s), _fmt(ela_s), _fmt(float(elb_h)), _fmt(float(ela_h)))


    i = 0
    while (data_zod[i, 0] < ela_h):
        i = i + 1

    if (i > 0):

        if ((data_zod[i,0] - ela_h) > (ela_h - data_zod[i-1,0])):
            i = i - 1

    j = 0
    while (data_zod[0,j] < elb_h):
        j = j + 1

    if (j > 0):
        if ((data_zod[0,j] - elb_h) > (elb_h - data_zod[0,j-1])):
            j = j - 1


    zod_value = data_zod[i,j]
    logging.info("zod_value=" + f"{zod_value:24.16E}".strip())

    wl_sol = np.asarray(channel.zod_spectrum_wavelength, dtype=np.float64)
    flux_sol = np.asarray(channel.zod_spectrum_flux, dtype=np.float64)
    zod_spectrum = zod_value * flux_sol
    zod_spectrum = np.asarray(zod_spectrum, dtype=np.float64)

    spline = CubicSpline(wl_sol, zod_spectrum, extrapolate=True)
    background = spline(channel.effective_area_wavelength)

    background*= channel.sky_pixel_area_arcsec2 / ARCSEC2_PER_SR 

    return background