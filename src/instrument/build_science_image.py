import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from instrument.bias_image import generate_bias_image

def build_science_images (spectra_2d_nuv, spectra_2d_vis, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, ctx: RunContext):

    nuv_img = build_science_image(spectra_2d_nuv, nuv, ctx)
    vis_img = build_science_image(spectra_2d_vis, vis, ctx)
    return nuv_img, vis_img


def build_science_image(spectra_2d, channel: SpectroscopyChannel, ctx: RunContext):
    nx = channel.x_pixels
    ny = channel.y_pixels
    image = np.zeros((ny, nx))

    bias = generate_bias_image(channel)
    image += bias
    ctx.write_image_png.write_image(image, "BIAS", ctx, channel)

    dark = 