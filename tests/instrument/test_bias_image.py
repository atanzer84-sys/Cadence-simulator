import numpy as np
import pytest
from unittest.mock import patch

from instrument.bias_image import generate_bias_image


# Tests: test_generate_bias_image_shape_and_stats
# Behavior: Bias image matches configured mean and noise scale
def test_generate_bias_image_shape_and_stats(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(bias_offset=42.0, read_noise=3.0)

    with patch('instrument.bias_image._rng', np.random.default_rng(42)):
        bias = generate_bias_image(ch)

    assert bias.shape == (ch.y_pixels, ch.x_pixels)
    assert np.isclose(float(bias.mean()), ch.bias_offset, atol=1.0)
    assert np.isclose(float(bias.std()), ch.read_noise, atol=0.5)


# Tests: test_generate_bias_image_has_randomness_when_noise_configured
# Behavior: Nonzero read noise produces a non-constant bias image
def test_generate_bias_image_has_randomness_when_noise_configured(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(bias_offset=42.0, read_noise=3.0)

    bias = generate_bias_image(ch)

    assert bias.shape == (ch.y_pixels, ch.x_pixels)
    assert not np.all(bias == ch.bias_offset)


# Tests: test_generate_bias_image_two_calls_different
# Behavior: Two generated bias images are not identical
def test_generate_bias_image_two_calls_different(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(bias_offset=42.0, read_noise=3.0)

    bias1 = generate_bias_image(ch)
    bias2 = generate_bias_image(ch)

    assert not np.array_equal(bias1, bias2)


# Tests: test_generate_bias_image_zero_noise
# Behavior: Zero read noise produces constant bias image
def test_generate_bias_image_zero_noise(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(read_noise=0.0, bias_offset=42.0)

    bias = generate_bias_image(ch)

    assert np.all(bias == 42.0)


# Tests: test_generate_bias_image_dtype
# Behavior: Output dtype is float32 and values are finite
def test_generate_bias_image_dtype(make_spectroscopy_channel):
    ch = make_spectroscopy_channel()

    bias = generate_bias_image(ch)

    assert bias.dtype == np.float32
    assert np.isfinite(bias).all()


# Tests: test_generate_bias_image_rejects_negative_read_noise
# Behavior: Negative read noise is invalid and raises
def test_generate_bias_image_rejects_negative_read_noise(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(read_noise=-1.0)

    with pytest.raises(ValueError):
        generate_bias_image(ch)


# Tests: test_generate_bias_image_rejects_non_numeric_bias_offset
# Behavior: Non-numeric bias offset is invalid and raises
def test_generate_bias_image_rejects_non_numeric_bias_offset(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(bias_offset="not-a-number")

    with pytest.raises(TypeError):
        generate_bias_image(ch)