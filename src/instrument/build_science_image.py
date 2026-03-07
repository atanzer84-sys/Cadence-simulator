import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from instrument.bias_image import generate_bias_image
from instrument.dark_image import generate_dark_image
from instrument.cosmic_image import generate_cosmic_rays
from configs.global_config import GlobalConfig, get_global_config
from instrument.background_image import generate_background_image
from domain.star import Star
from domain.star_catalog import StarCatalog
from instrument.background_star_preparation import populate_background_star_catalog
from instrument.spectrum_spread import get_spectrum_placement, smear_1d_spectrum_dispersion, spread_1d_spectrum_to_2d
from instrument.photon_noise import apply_photon_noise_gauss_from_spectra2d



def build_science_images(spectra_2d_nuv, spectra_2d_vis, rate_nir, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, star: Star):
    cfg = get_global_config()
    background_stars_catalog = populate_background_star_catalog(nuv, vis, nir, ctx, cfg, star)

    for channel in (nuv, vis, nir):
        ctx.produce_plots.plot_background_star_counts(background_stars_catalog, channel, ctx)

    nuv_imgs = _create_spectroscopy_channel_images(spectra_2d_nuv, nuv, ctx, cfg, star, background_stars_catalog)
    # vis_imgs = _create_spectroscopy_channel_images(spectra_2d_vis, vis, ctx, cfg, star, background_stars_catalog)
    vis_imgs = []  # TODO: re-enable 

    print("\n==== STARTING SCIENCE IMAGE GENERATION (NIR) =====")
    # nir_img = build_science_image_photometry(rate_nir, nir, ctx, cfg, star, background_stars_catalog)
    nir_img = []  # TODO: re-enable

    return nuv_imgs, vis_imgs, nir_img

def _create_spectroscopy_channel_images(spectra_2d, channel: SpectroscopyChannel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog) -> list:
    print(f"\n==== STARTING SCIENCE IMAGE GENERATION ({channel.channel_name}) =====")
    logging.info("Science Image generation starting for channel %s", channel.channel_name)

    exposure = channel.exposure_s
    # compute artifacts that do not change from one exposure to the next to save time.
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    spectra_component = spectra_2d * exposure
    background_component = generate_background_image(channel, ctx, star)
    images = []

    for frame_index in range(channel.n_science_frames):
        time_s = frame_index * (channel.exposure_s + cfg.readout_gap_s)
        roll_angle_deg = 360.0 * (time_s / orbit_duration_s)

        print(f"science exposure image {frame_index + 1}/{channel.n_science_frames} (roll_angle={roll_angle_deg:.2f}°)")
        logging.info("science exposure image: frame_index=%d n_science_frames=%d time_s=%g roll_angle_deg=%g", frame_index, channel.n_science_frames, time_s, roll_angle_deg)

        img = _create_spectroscopy_per_exposure(spectra_component, background_component, channel, ctx, cfg, star, background_stars_catalog, frame_index, roll_angle_deg)
        images.append(img)

    return images


def _create_spectroscopy_per_exposure(spectra_component, background_component, channel: SpectroscopyChannel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, frame_index: int, roll_angle_deg: float) -> np.ndarray:
    """Build one science frame. Constant debug images written once (frame 0); background stars use index (vary with roll)."""
    ccd_gain = channel.ccd_gain

    image = np.zeros((channel.y_pixels, channel.x_pixels))
    img_spectra_bgstars = np.zeros((channel.y_pixels, channel.x_pixels))

    bias = generate_bias_image(channel)
    image += bias
    if frame_index < 1:
        ctx.write_image_png.write_image_png(image, "SCIENCE_BIAS_ONLY", ctx, channel, star=star, index=frame_index)

    dark = generate_dark_image(channel)
    image += dark
    if frame_index < 1:
        ctx.write_image_png.write_image_png(image, "SCIENCE_DARK_ONLY", ctx, channel, star=star, index=frame_index)

    image += spectra_component
    img_spectra_bgstars += spectra_component
    if frame_index < 1:
        ctx.write_image_png.write_image_png(spectra_component, "SCIENCE_SIGNAL_ONLY", ctx, channel, star=star, index=frame_index)
        ctx.write_image_png.write_image_png(image, "SCIENCE_SPECTRA", ctx, channel, star=star, index=frame_index)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(spectra_component, channel, ctx, star)
    image += photon_noise
    if frame_index < 1:
        ctx.write_image_png.write_image_png(photon_noise, "SCIENCE_PHOTON_NOISE_ONLY", ctx, channel, star=star, index=frame_index)

    image += background_component
    if frame_index < 1:
        ctx.write_image_png.write_image_png(image, "SCIENCE_BACKGROUND_ONLY", ctx, channel, index=frame_index)

    bg_stars = _create_spectroscopy_per_roll_angle(channel, ctx, star, background_stars_catalog, roll_angle_deg, frame_index)
    image += bg_stars
    img_spectra_bgstars += bg_stars

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    if frame_index < 1:
        ctx.write_image_png.write_image_png(cosmic, "SCIENCE_COSMIC_ONLY", ctx, channel, star=star, index=frame_index)

    image = image * ccd_gain
    img_spectra_bgstars = img_spectra_bgstars * ccd_gain
    ctx.write_image_png.write_image_png(image, "SCIENCE_COMPLETELY_MERGED", ctx, channel, star=star, index=frame_index)
    ctx.write_background_star_png.write_background_star_visibility_tests(image, bg_stars, "SCIENCE_BG_VISIBILITY", ctx, channel, star=star, index=frame_index)

    return image


def _create_spectroscopy_per_roll_angle(channel: SpectroscopyChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog, roll_angle_deg: float, frame_index: int) -> np.ndarray:
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float64)
    x_target_star, y_target_star, slope, intercept = get_spectrum_placement(channel)
    cos_roll_angle, sine_roll_angle, half_width_slit, half_length_slit = _prepare_slit_geometry(channel, roll_angle_deg)

    total = len(background_stars_catalog.stars_by_id)
    n_in_slit = 0
    for star_id in background_stars_catalog.stars_by_id:
        bg_star_2d = _add_background_star_if_in_slit(star_id, channel, ctx, background_stars_catalog, frame_index, x_target_star, y_target_star, slope, intercept, cos_roll_angle, sine_roll_angle, half_width_slit, half_length_slit)
        if bg_star_2d is not None:
            image += bg_star_2d
            n_in_slit += 1

    logging.info("BG STARS roll_angle: frame=%d channel=%s roll_angle_deg=%g n_in_slit=%d/%d image_sum=%g", frame_index, channel.channel_name, float(roll_angle_deg), int(n_in_slit), int(total), float(np.sum(image)))
    
    return image


def _add_background_star_if_in_slit(star_id: str, channel: SpectroscopyChannel, ctx: RunContext, background_stars_catalog: StarCatalog, frame_index: int, x_target_star: int, y_target_star: int, slope: float, intercept: float, cos_roll_angle: float, sine_roll_angle: float, half_width_slit: float, half_length_slit: float) -> np.ndarray | None:
    """Return the 2d background star image to add, or None if star is outside slit or has no counts."""
    dx, dy = background_stars_catalog.get_offset_arcsec(star_id)
    horizontal_relative_position, vertical_relative_position, inside_slit = _is_background_star_inside_slit(dx, dy, cos_roll_angle, sine_roll_angle, half_width_slit, half_length_slit)

    if not inside_slit:
        return None

    key = (star_id, channel.channel_name)
    if key not in background_stars_catalog.counts_by_id_and_band:
        logging.info("BG STAR missing cached counts: frame=%d channel=%s star_id=%s", frame_index, channel.channel_name, star_id)
        return None

    counts_s_px = background_stars_catalog.counts_by_id_and_band[key]
    y_background_star = _get_background_star_detector_row(y_target_star, vertical_relative_position, channel)
    counts_smeared_px = smear_1d_spectrum_dispersion(counts_s_px, channel)
    background_star_2d = spread_1d_spectrum_to_2d(counts_smeared_px, channel, x_target_star, float(y_background_star), slope, intercept, announce_user=False)

    bg_star = background_stars_catalog.stars_by_id[star_id]
    formatted = f"{int(star_id.split('_')[1]):,}".replace(",", " ")
    mag = bg_star.gaia_magnitude if bg_star.gaia_magnitude is not None else float("nan")
    ra = bg_star.right_ascension if bg_star.right_ascension is not None else float("nan")
    dec = bg_star.declination if bg_star.declination is not None else float("nan")
    logging.info("BG STAR in slit: frame=%d channel=%s star_id_formatted=%s star_id=%s mag=%.3f dx=%g dy=%g u=%g v=%g half_w=%g half_l=%g ra=%.6f dec=%.6f detector_y=%.2f x=%d", frame_index, channel.channel_name, formatted, star_id, mag, dx, dy, horizontal_relative_position, vertical_relative_position, half_width_slit, half_length_slit, ra, dec, float(y_background_star), x_target_star)

    return background_star_2d


def _prepare_slit_geometry(channel: SpectroscopyChannel, roll_angle_deg: float):
    roll_angle_radians = np.deg2rad(float(roll_angle_deg))
    cos_roll_angle = float(np.cos(roll_angle_radians))
    sin_roll_angle = float(np.sin(roll_angle_radians))
    half_width_slit = float(channel.slit_half_width_arcsec)
    half_length_slit = float(channel.slit_half_length_arcsec)
    return cos_roll_angle, sin_roll_angle, half_width_slit, half_length_slit

def _is_background_star_inside_slit(dx: float, dy: float, cos_roll_angle: float, sin_roll_angle: float, half_width_slit: float, half_length_slit: float):
    u = dx * cos_roll_angle + dy * sin_roll_angle
    v = -dx * sin_roll_angle + dy * cos_roll_angle
    inside_slit = abs(u) <= half_width_slit and abs(v) <= half_length_slit
    
    return u, v, inside_slit

def _get_background_star_detector_row(y0: int, vertical_relative_position: float, channel: SpectroscopyChannel) -> int:
    y_offset_pix = vertical_relative_position / channel.pixel_scale
    y_background_star = int(round(y0 + y_offset_pix))
    logging.info("BG star row placement: target star (y0) y position: %d, background star y position: %d", y0, y_background_star)
    return y_background_star


def build_science_image_photometry(nir_rate, channel: PhotometryChannel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog):
    logging.info("Science Image generation starting for channel %s", channel.channel_name)
    nx = channel.x_pixels
    ny = channel.y_pixels
    image = np.zeros((ny, nx))
    exposure = channel.exposure_s
    ccd_gain = channel.ccd_gain

    bias = generate_bias_image(channel)
    image += bias
    ctx.write_image_png.write_image_png(image, "SCIENCE_BIAS", ctx, channel, star=star)

    dark = generate_dark_image(channel)
    image += dark
    ctx.write_image_png.write_image_png(image, "SCIENCE_DARK", ctx, channel, star=star)

    spectra = nir_rate * exposure
    ctx.write_image_png.write_image_png(spectra, "SIGNAL_ONLY", ctx, channel, star=star)
    image += spectra
    ctx.write_image_png.write_image_png(image, "SCIENCE_SPECTRA", ctx, channel, star=star)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(nir_rate*exposure, channel, ctx, star)
    image += photon_noise
    ctx.write_image_png.write_image_png(image, "SCIENCE_PHOTON_NOISE", ctx, channel, star=star)

    background = generate_background_image(channel, ctx, star)
    image += background
    ctx.write_image_png.write_image_png(image, "SCIENCE_BACKGROUND", ctx, channel)
    
    # TODO BACKGROUND STAR GENERATION

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    ctx.write_image_png.write_image_png(image, "SCIENCE_COSMIC", ctx, channel, star=star)
    

    image = image * ccd_gain
    ctx.write_image_png.write_image_png(image, "SCIENCE_COMPLETELY_MERGED", ctx, channel, star=star)

    return image