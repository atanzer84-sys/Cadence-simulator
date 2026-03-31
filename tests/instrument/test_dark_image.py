import numpy as np
from unittest.mock import patch

from instrument.dark_image import generate_dark_image


# Tests: test_generate_dark_image_shape_and_finite_values
# Behavior: Dark image has correct shape and finite values
def test_generate_dark_image_shape_and_finite_values(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.5,
        exposure_s=10.0,
    )

    with patch('instrument.dark_image._rng', np.random.default_rng(42)):
        dark = generate_dark_image(ch)

    assert dark.shape == (ch.y_pixels, ch.x_pixels)
    assert np.isfinite(dark).all()


# Tests: test_generate_dark_image_has_randomness_when_configured
# Behavior: Nonzero sigma produces a non-constant dark image
def test_generate_dark_image_has_randomness_when_configured(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.5,
        exposure_s=10.0,
    )

    dark = generate_dark_image(ch)

    assert dark.shape == (ch.y_pixels, ch.x_pixels)
    assert not np.all(dark == dark[0, 0])


# Tests: test_generate_dark_image_two_calls_different
# Behavior: Two generated dark images are not identical
def test_generate_dark_image_two_calls_different(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.5,
        exposure_s=10.0,
    )

    dark1 = generate_dark_image(ch)
    dark2 = generate_dark_image(ch)

    assert not np.array_equal(dark1, dark2)


# Tests: test_generate_dark_image_noise_only
# Behavior: Zero dark_noise still produces variation if sigma is nonzero
def test_generate_dark_image_noise_only(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=0.0,
        dark_current_noise=1.0,
        exposure_s=10.0,
    )

    dark1 = generate_dark_image(ch)
    dark2 = generate_dark_image(ch)

    assert not np.array_equal(dark1, dark2)


# Tests: test_generate_dark_image_deterministic_when_sigma_zero
# Behavior: Zero sigma produces deterministic output
def test_generate_dark_image_deterministic_when_sigma_zero(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.0,
        exposure_s=10.0,
    )

    dark1 = generate_dark_image(ch)
    dark2 = generate_dark_image(ch)

    assert np.array_equal(dark1, dark2)


# Tests: test_generate_dark_image_exposure_zero
# Behavior: Zero exposure yields base distribution only
def test_generate_dark_image_exposure_zero(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.5,
        exposure_s=0.0,
    )

    with patch('instrument.dark_image._rng', np.random.default_rng(42)):
        dark = generate_dark_image(ch)

    rng = np.random.default_rng(42)
    expected = rng.normal(1.0, 0.5, size=(ch.y_pixels, ch.x_pixels)).astype(np.float32)

    assert np.allclose(dark, expected)

# Tests: test_generate_dark_image_dtype
# Behavior: Output dtype is float32
def test_generate_dark_image_dtype(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        dark_current=1.0,
        dark_current_noise=0.5,
    )

    dark = generate_dark_image(ch)

    assert dark.dtype == np.float32