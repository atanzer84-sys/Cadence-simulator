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
from instrument.background_star_spectroscopy import generate_background_star_spectroscopy_image
from instrument.photon_noise import apply_photon_noise_gauss_from_spectra2d



def build_science_images(spectra_2d_nuv, spectra_2d_vis, rate_nir, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, star: Star):
    cfg = get_global_config()
    background_stars_catalog = populate_background_star_catalog(nuv, vis, nir, ctx, cfg, star)

    for channel in (nuv, vis, nir):
        ctx.produce_plots.plot_background_star_counts(background_stars_catalog, channel, ctx, star)

    nuv_imgs = _create_spectroscopy_channel_images(spectra_2d_nuv, nuv, ctx, cfg, star, background_stars_catalog)
    vis_imgs = _create_spectroscopy_channel_images(spectra_2d_vis, vis, ctx, cfg, star, background_stars_catalog)
    # vis_imgs = []  # TODO: re-enable 

    print("\n==== STARTING SCIENCE IMAGE GENERATION (NIR) =====")
    nir_img = build_science_image_photometry(rate_nir, nir, ctx, cfg, star, background_stars_catalog)
    # nir_img = []  # TODO: re-enable

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
        ctx.write_image_png.write_image_png(background_component, "SCIENCE_BACKGROUND_ONLY", ctx, channel, index=frame_index)

    bg_stars, background_star_bands = generate_background_star_spectroscopy_image(channel, ctx, star, background_stars_catalog, roll_angle_deg, frame_index)
    image += bg_stars
    img_spectra_bgstars += bg_stars

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    if frame_index < 1:
        ctx.write_image_png.write_image_png(cosmic, "SCIENCE_COSMIC_ONLY", ctx, channel, star=star, index=frame_index)

    image = image * ccd_gain
    img_spectra_bgstars = img_spectra_bgstars * ccd_gain
    ctx.write_image_png.write_image_png(image, "SCIENCE_COMPLETELY_MERGED", ctx, channel, star=star, index=frame_index)
    ctx.write_background_star_png.write_background_star_visibility_tests(image, bg_stars, "SCIENCE PANEL", ctx, channel, star=star, index=frame_index, background_star_bands=background_star_bands)

    return image


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