import numpy as np

from instrument.bias_image import generate_bias_image


def test_generate_bias_image_shape_and_mean(make_channel):
    np.random.seed(0)
    ch = make_channel()

    bias = generate_bias_image(ch)

    assert bias.shape == (ch.y_pixels, ch.x_pixels)

    mean_val = float(bias.mean())
    # Roughly around bias_offset, allow for noise
    assert 90.0 < mean_val < 110.0

