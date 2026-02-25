import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from instrument.bias_image import generate_bias_image
from instrument.dark_image import generate_dark_image

def build_science_images (spectra_2d_nuv, spectra_2d_vis, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, ctx: RunContext):

    nuv_img = build_science_image(spectra_2d_nuv, nuv, ctx)
    vis_img = build_science_image(spectra_2d_vis, vis, ctx)
    return nuv_img, vis_img


def build_science_image(spectra_2d, channel: SpectroscopyChannel, ctx: RunContext):
    nx = channel.x_pixels
    ny = channel.y_pixels
    image = np.zeros((ny, nx))
    exposure = channel.exposure_s
    ccd_gain = channel.ccd_gain

    bias = generate_bias_image(channel)
    image += bias
    ctx.write_image_png.write_image(image, "BIAS", ctx, channel)

    dark = generate_dark_image(channel)
    image += dark
    ctx.write_image_png.write_image(image, "DARK", ctx, channel)

    spectra = spectra_2d * exposure
    image += spectra
    ctx.write_image_png.write_image(image, "SPECTRA", ctx, channel)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(spectra_2d*exposure)
    image += photon_noise
    ctx.write_image_png.write_image(image, "PHOTON_NOISE", ctx, channel)

    image = image * ccd_gain
    return image

def apply_photon_noise_gauss_from_spectra2d(spectra_2d_exposure):
    
    distr = np.random.normal(loc=0.0, scale=1.0, size=spectra_2d_exposure.shape)
    sigma = np.sqrt(np.clip(spectra_2d_exposure, 0, None))
    noise = distr * sigma
    return noise