import numpy as np
from types import SimpleNamespace

from instrument.dark_image import generate_dark_image


def _channel():
    # Minimal SpectroscopyChannel-like object for dark_image tests
    return SimpleNamespace(
        channel_name="NUV",
        x_pixels=10,
        y_pixels=8,
        dark_noise=0.5,
        dark_current_sigma=2.0,
        exposure_s=10.0,
    )


def test_generate_dark_image_shape_and_mean_positive():
    np.random.seed(0)
    ch = _channel()

    dark = generate_dark_image(ch)

    assert dark.shape == (ch.y_pixels, ch.x_pixels)
    # Dark should be positive on average due to dark current contribution
    assert float(dark.mean()) > 0.0

