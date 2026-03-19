import numpy as np
import pytest

from instrument.bias_image import generate_bias_image


# Tests: generate_bias_image
# Behavior: returns correct shape and deterministic statistics with fixed seed
def test_generate_bias_image_shape_and_stats(make_spectroscopy_channel):
    np.random.seed(0)
    ch = make_spectroscopy_channel()

    bias = generate_bias_image(ch)

    assert bias.shape == (ch.y_pixels, ch.x_pixels)

    assert np.isclose(float(bias.mean()), ch.bias_offset, atol=5.0)
    assert np.isclose(float(bias.std()), ch.read_noise, atol=2.0)


# Tests: generate_bias_image
# Behavior: zero read noise produces constant bias image
def test_generate_bias_image_zero_noise(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(read_noise=0.0, bias_offset=42.0)

    bias = generate_bias_image(ch)

    assert np.all(bias == 42.0)


# Tests: generate_bias_image
# Behavior: output dtype is float32
def test_generate_bias_image_dtype(make_spectroscopy_channel):
    ch = make_spectroscopy_channel()

    bias = generate_bias_image(ch)

    assert bias.dtype == np.float32


# Tests: generate_bias_image
# Behavior: negative read noise is invalid and raises
def test_generate_bias_image_rejects_negative_read_noise(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(read_noise=-1.0)

    with pytest.raises(ValueError):
        generate_bias_image(ch)


# Tests: generate_bias_image
# Behavior: non-numeric bias offset is invalid and raises
def test_generate_bias_image_rejects_non_numeric_bias_offset(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(bias_offset="not-a-number")

    with pytest.raises(TypeError):
        generate_bias_image(ch)