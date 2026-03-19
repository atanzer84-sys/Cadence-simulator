import numpy as np

from instrument.dark_image import generate_dark_image


def test_generate_dark_image_shape_and_mean_positive(make_channel):
    np.random.seed(0)
    ch = make_channel()

    dark = generate_dark_image(ch)

    assert dark.shape == (ch.y_pixels, ch.x_pixels)
    # Dark should be positive on average due to dark current contribution
    assert float(dark.mean()) > 0.0

