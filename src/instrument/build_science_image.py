import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel, PhotometryChannel, Channel
from instrument.bias_image import generate_bias_image
from instrument.dark_image import generate_dark_image
from instrument.cosmic_image import generate_cosmic_rays
from configs.global_config import GlobalConfig, get_global_config
from instrument.background_image import generate_background_image
from instrument.background_stars_image import generate_background_stars_image
from domain.star import Star
from domain.star_catalog import StarCatalog
from loaders.load_background_stars import lookup_background_stars

def build_science_images (spectra_2d_nuv, spectra_2d_vis, rate_nir, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, star: Star):

    cfg = get_global_config()

    background_stars_catalog = lookup_background_stars(ctx, cfg, star)

    print("\n==== STARTING SCIENCE IMAGE GENERATION (NUV & VIS) =====")
    nuv_img = build_science_image(spectra_2d_nuv, nuv, ctx, cfg, star, background_stars_catalog)
    vis_img = build_science_image(spectra_2d_vis, vis, ctx, cfg, star, background_stars_catalog)

    print("\n==== STARTING SCIENCE IMAGE GENERATION (NIR) =====")
    nir_img = build_science_image(rate_nir, vis, ctx, cfg, star, background_stars_catalog)

    return nuv_img, vis_img, nir_img


def build_science_image(spectra_2d, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog):
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

    spectra = spectra_2d * exposure
    ctx.write_image_png.write_image(spectra, "SIGNAL_ONLY", ctx, channel, star=star)
    image += spectra
    ctx.write_image_png.write_image(image, "SCIENCE_SPECTRA", ctx, channel, star=star)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(spectra_2d*exposure, channel, ctx, star)
    image += photon_noise
    ctx.write_image_png.write_image(image, "SCIENCE_PHOTON_NOISE", ctx, channel, star=star)

    background = generate_background_image(channel, ctx, star)
    image += background
    ctx.write_image_png.write_image(image, "SCIENCE_BACKGROUND", ctx, channel)
    
    generate_background_stars_image(channel, ctx, star, background_stars_catalog)


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
    ctx.write_image_png.write_image(noise, "NOISE_ONLY", ctx, channel, star=star)

    return noise