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


def build_science_images(spectra_2d_nuv, spectra_2d_vis, rate_nir, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, star: Star):
    cfg = get_global_config()
    background_stars_catalog = populate_background_star_catalog(nuv, vis, nir, ctx, cfg, star)

    nuv_imgs = _create_spectroscopy_channel_images(spectra_2d_nuv, nuv, ctx, cfg, star, background_stars_catalog)
    # vis_imgs = _create_spectroscopy_channel_images(spectra_2d_vis, vis, ctx, cfg, star, background_stars_catalog)

    # print("\n==== STARTING SCIENCE IMAGE GENERATION (NIR) =====")
    # nir_img = build_science_image_photometry(rate_nir, nir, ctx, cfg, star, background_stars_catalog)

    # return nuv_imgs, vis_imgs, nir_img
    return nuv_imgs

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

        # TODO: THIS IS A MESS. too many params and i need an index for the images
        img = _create_spectroscopy_per_exposure(spectra_component, background_component, channel, ctx, cfg, star, background_stars_catalog, frame_index, roll_angle_deg)
        images.append(img)

    return images


def _create_spectroscopy_per_exposure(spectra_component, background_component, channel: SpectroscopyChannel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, frame_index: int, roll_angle_deg: float) -> np.ndarray:
    """Build one science frame. Constant debug images written once (frame 0); background stars use index (vary with roll)."""
    nx = channel.x_pixels
    ny = channel.y_pixels
    ccd_gain = channel.ccd_gain

    image = np.zeros((ny, nx))

    bias = generate_bias_image(channel)
    image += bias
    ctx.write_image_png.write_image(image, "SCIENCE_BIAS", ctx, channel, star=star, index=frame_index)

    dark = generate_dark_image(channel)
    image += dark
    ctx.write_image_png.write_image(image, "SCIENCE_DARK", ctx, channel, star=star, index=frame_index)

    image += spectra_component
    ctx.write_image_png.write_image(spectra_component, "SIGNAL_ONLY", ctx, channel, star=star, index=frame_index)
    ctx.write_image_png.write_image(image, "SCIENCE_SPECTRA", ctx, channel, star=star, index=frame_index)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(spectra_component, channel, ctx, star)
    image += photon_noise
    ctx.write_image_png.write_image(image, "SCIENCE_PHOTON_NOISE", ctx, channel, star=star, index=frame_index)

    image += background_component
    ctx.write_image_png.write_image(image, "SCIENCE_BACKGROUND", ctx, channel, index=frame_index)

    bg_stars = _create_spectroscopy_per_roll_angle(channel, ctx, star, background_stars_catalog, roll_angle_deg, frame_index)
    image += bg_stars
    ctx.write_image_png.write_image(image, "SCIENCE_BACKGROUND_STARS", ctx, channel, star=star, index=frame_index)

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    ctx.write_image_png.write_image(image, "SCIENCE_COSMIC", ctx, channel, star=star, index=frame_index)

    image = image * ccd_gain
    ctx.write_image_png.write_image(image, "SCIENCE_COMPLETELY_MERGED", ctx, channel, star=star, index=frame_index)

    return image


def _create_spectroscopy_per_roll_angle(channel: SpectroscopyChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog, roll_angle_deg: float, frame_index: int) -> np.ndarray:
    
    nx = channel.x_pixels
    ny = channel.y_pixels
    exposure_s = channel.exposure_s
    image = np.zeros((ny, nx))

    cos_roll_angle, sine_roll_angle, half_width_slit, half_length_slit = _prepare_slit_geometry(channel, roll_angle_deg)

    y0 = int(np.clip(int(round(float(channel.intercept_pixels))), 0, ny - 1))
    
    total = len(background_stars_catalog.stars_by_id)
    inside = 0
    for star_id in background_stars_catalog.stars_by_id:
        dx, dy = background_stars_catalog.get_offset_arcsec(star_id)
        u, v, inside_slit = _is_background_star_inside_slit(dx, dy, cos_roll_angle, sine_roll_angle, half_width_slit, half_length_slit)
        
        #TODO: REMOVE
        formatted = f"{int(star_id.split('_')[1]):,}".replace(",", " ")
        logging.info("BG STAR slit check: frame=%d channel=%s star_id=%s dx=%g dy=%g u=%g v=%g half_w=%g half_l=%g inside=%s", frame_index, channel.channel_name, formatted, dx, dy, u, v, half_width_slit, half_length_slit, inside_slit)

        if not inside_slit:
            continue

        key = (star_id, channel.channel_name)
        if key not in background_stars_catalog.counts_by_id_and_band:
            logging.info("BG STAR missing cached counts: frame=%d channel=%s star_id=%s", frame_index, channel.channel_name, star_id)
            continue

        counts_s_px = background_stars_catalog.counts_by_id_and_band[key]
        # TODO: REMOVE
        logging.info("BG STAR accepted: frame=%d channel=%s star_id=%s sum=%g max=%g", frame_index, channel.channel_name, formatted, float(np.sum(counts_s_px * float(exposure_s))), float(np.max(counts_s_px * float(exposure_s))))

        _add_background_star_placeholder_contamination(image, y0, counts_s_px, channel.crossing_time_s)

        inside += 1

    ctx.write_image_png.write_image(image, "SCIENCE_BACKGROUND_STARS_ONLY", ctx, channel, star=star, index=frame_index)

    logging.info("BG STARS roll_angle: channel=%s roll_angle_deg=%g inside=%d/%d sum=%g", channel.channel_name, float(roll_angle_deg), int(inside), int(total), float(np.sum(image)))
    return image

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

def _add_background_star_placeholder_contamination(image: np.ndarray, y0: int, counts_s_px: np.ndarray, crossing_time_s: float):
    """Mutate image in place: add counts to row y0."""
    image[y0, :] += counts_s_px * crossing_time_s

def build_science_image_photometry(nir_rate, channel: PhotometryChannel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog):
    logging.info("Science Image generation starting for channel %s", channel.channel_name)
    nx = channel.x_pixels
    ny = channel.y_pixels
    image = np.zeros((ny, nx))
    exposure = channel.exposure_s
    ccd_gain = channel.ccd_gain

    bias = generate_bias_image(channel)
    image += bias
    ctx.write_image_png.write_image(image, "SCIENCE_BIAS", ctx, channel, star=star)

    dark = generate_dark_image(channel)
    image += dark
    ctx.write_image_png.write_image(image, "SCIENCE_DARK", ctx, channel, star=star)

    spectra = nir_rate * exposure
    ctx.write_image_png.write_image(spectra, "SIGNAL_ONLY", ctx, channel, star=star)
    image += spectra
    ctx.write_image_png.write_image(image, "SCIENCE_SPECTRA", ctx, channel, star=star)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(nir_rate*exposure, channel, ctx, star)
    image += photon_noise
    ctx.write_image_png.write_image(image, "SCIENCE_PHOTON_NOISE", ctx, channel, star=star)

    background = generate_background_image(channel, ctx, star)
    image += background
    ctx.write_image_png.write_image(image, "SCIENCE_BACKGROUND", ctx, channel)
    
    # TODO BACKGROUND STAR GENERATION

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    ctx.write_image_png.write_image(image, "SCIENCE_COSMIC", ctx, channel, star=star)
    

    image = image * ccd_gain
    ctx.write_image_png.write_image(image, "SCIENCE_COMPLETELY_MERGED", ctx, channel, star=star)

    return image
    

def apply_photon_noise_gauss_from_spectra2d(spectra_2d_exposure, channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    distr = np.random.normal(loc=0.0, scale=1.0, size=spectra_2d_exposure.shape)
    sigma = np.sqrt(np.clip(spectra_2d_exposure, 0, None))
    noise = distr * sigma
    ctx.write_image_png.write_image(noise, "NOISE_ONLY", ctx, channel, star=star, index=0)

    return noise