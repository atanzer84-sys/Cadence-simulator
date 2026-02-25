import numpy as np

from types import SimpleNamespace

from instrument.bias_image import generate_bias_image


def _channel():
    # Minimal SpectroscopyChannel-like object for bias_image tests
    return SimpleNamespace(
        channel_name="NUV",
        x_pixels=10,
        y_pixels=8,
        bias_offset=100.0,
        read_noise=5.0,
        ccd_gain=2.0,
    )


def test_generate_bias_image_shape_and_mean():
    np.random.seed(0)
    ch = _channel()

    bias = generate_bias_image(ch)

    assert bias.shape == (ch.y_pixels, ch.x_pixels)

    mean_val = float(bias.mean())
    # Roughly around bias_offset, allow for noise
    assert 90.0 < mean_val < 110.0

