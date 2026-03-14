import numpy as np

from instrument.dark_image import generate_dark_image
from tests.helpers.channel_factory import channel


def test_generate_dark_image_shape_and_mean_positive():
    np.random.seed(0)
    ch = channel()

    dark = generate_dark_image(ch)

    assert dark.shape == (ch.y_pixels, ch.x_pixels)
    # Dark should be positive on average due to dark current contribution
    assert float(dark.mean()) > 0.0

