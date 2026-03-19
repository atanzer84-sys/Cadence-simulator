import numpy as np
from unittest.mock import patch

from instrument.photon_noise import generate_photon_noise_from_spectra2d


# Tests: test_generate_photon_noise_shape_dtype_and_finite
# Behavior: Noise output preserves shape, dtype, and finite values
def test_generate_photon_noise_shape_dtype_and_finite():
    spectra_2d_exposure = np.array([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=np.float32)

    with patch('instrument.photon_noise._rng', np.random.default_rng(42)):
        noise = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    assert noise.shape == spectra_2d_exposure.shape
    assert noise.dtype == np.float32
    assert np.isfinite(noise).all()


# Tests: test_generate_photon_noise_two_calls_different
# Behavior: Two generated noise arrays are not identical
def test_generate_photon_noise_two_calls_different():
    spectra_2d_exposure = np.full((4, 5), 9.0, dtype=np.float32)

    noise1 = generate_photon_noise_from_spectra2d(spectra_2d_exposure)
    noise2 = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    assert not np.array_equal(noise1, noise2)


# Tests: test_generate_photon_noise_zero_exposure
# Behavior: Zero exposure produces zero noise everywhere
def test_generate_photon_noise_zero_exposure():
    spectra_2d_exposure = np.zeros((3, 4), dtype=np.float32)

    noise = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    assert np.all(noise == 0.0)


# Tests: test_generate_photon_noise_negative_values_clipped_to_zero
# Behavior: Negative exposure values produce zero noise
def test_generate_photon_noise_negative_values_clipped_to_zero():
    spectra_2d_exposure = np.array([[-1.0, -4.0], [-9.0, -16.0]], dtype=np.float32)

    noise = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    assert np.all(noise == 0.0)


# Tests: test_generate_photon_noise_matches_expected_with_fixed_rng
# Behavior: Patched RNG yields deterministic noise values
def test_generate_photon_noise_matches_expected_with_fixed_rng():
    spectra_2d_exposure = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)

    with patch('instrument.photon_noise._rng', np.random.default_rng(42)):
        noise = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    rng = np.random.default_rng(42)
    expected_distr = rng.normal(loc=0.0, scale=1.0, size=spectra_2d_exposure.shape).astype(np.float32)
    expected_sigma = np.sqrt(np.clip(spectra_2d_exposure, 0, None))
    expected_noise = expected_distr * expected_sigma

    assert np.allclose(noise, expected_noise)


# Tests: test_generate_photon_noise_zero_sigma_rows_deterministic
# Behavior: Zero sigma positions remain zero regardless of random draw
def test_generate_photon_noise_zero_sigma_rows_deterministic():
    spectra_2d_exposure = np.array([[0.0, 0.0], [1.0, 4.0]], dtype=np.float32)

    with patch('instrument.photon_noise._rng', np.random.default_rng(42)):
        noise = generate_photon_noise_from_spectra2d(spectra_2d_exposure)

    assert np.all(noise[0] == 0.0)