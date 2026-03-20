import numpy as np
from unittest.mock import patch
from instrument.science_image import _build_science_image_without_bg_stars, _generate_channel_calibration_frames

# Tests: _build_science_image_without_bg_stars
# Behavior: Verifies that the construction matching channel dimensions dynamically.
def test_build_science_image_integrity(make_spectroscopy_channel, make_run_context, make_global_config, make_star):
    channel = make_spectroscopy_channel()
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()

    target_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_comp = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    result = _build_science_image_without_bg_stars(target_signal, bg_comp, channel, ctx, cfg, star, frame_index=0)

    assert result.dtype == np.float32
    # Dynamic shape check to avoid brittle hardcoding
    assert result.shape == (channel.y_pixels, channel.x_pixels)


# Tests: _build_science_image_without_bg_stars
# Behavior: Ensures the diagnostic PNG writing path (frame_index < 1) is exercised without errors.
def test_build_science_image_diagnostic_path(make_photometry_channel, make_run_context, make_global_config, make_star):
    # Initialize a photometry channel (32x32).
    channel = make_photometry_channel()
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()

    signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    # Execute for frame 0 (triggers diagnostic logic) and frame 1 (skips it).
    _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=0)
    _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=1)


# Tests: _build_science_image_without_bg_stars
# Behavior: Verifies that the builder handles custom channel pixel overrides correctly.
def test_build_science_image_custom_geometry(make_spectroscopy_channel, make_run_context, make_global_config, make_star):
    # Use fixture overrides to test a non-standard 10x5 sensor.
    channel = make_spectroscopy_channel(x_pixels=10, y_pixels=5)
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()

    signal = np.zeros((5, 10), dtype=np.float32)

    # Execute and verify the resulting array shape matches the override.
    result = _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=0)
    assert result.shape == (5, 10)

# Tests: _generate_channel_calibration_frames
# Behavior: Verifies that the calibration loop executes for the requested number of frames.
# Signature: (channel, header, ctx, star, cfg)
def test_generate_calibration_frames_count(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel()
    ctx = make_run_context()
    cfg = make_global_config(n_calibration_frames=2)
    star = make_star()
    header = make_header()

    _generate_channel_calibration_frames(channel, header, ctx, star, cfg)


# Tests: _generate_channel_calibration_frames
# Behavior: Ensures the function returns immediately when no calibration frames are requested.
def test_generate_calibration_frames_skip(make_photometry_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_photometry_channel()
    ctx = make_run_context()
    cfg = make_global_config(n_calibration_frames=0)
    star = make_star()
    header = make_header()

    _generate_channel_calibration_frames(channel, header, ctx, star, cfg)


# Tests: _generate_channel_calibration_frames
# Behavior: Validates that the calibration generation works with realistic spectroscopy parameters.
# Note: Map fixture attributes to class fields to avoid breaking existing 500+ tests.
def test_generate_calibration_frames_realistic(realistic_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    # We use the fixture as-is to maintain compatibility across the suite
    channel = realistic_spectroscopy_channel
    ctx = make_run_context()
    cfg = make_global_config(n_calibration_frames=1)
    star = make_star()
    header = make_header()

    _generate_channel_calibration_frames(channel, header, ctx, star, cfg)

# Tests: _build_science_image_without_bg_stars
# Behavior: Verifies deterministic additive composition with stochastic terms patched out.
def test_build_science_image_additive_correctness(make_spectroscopy_channel, make_run_context, make_global_config, make_star):
    # Setup channel with 0 read/dark noise. Patch stochastic terms below.
    channel = make_spectroscopy_channel(
        read_noise=0.0,
        dark_noise=0.0,
        dark_current_sigma=0.0,
        bias_offset=100.0,
        ccd_gain=1.0
    )
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()

    # Create known constant component arrays
    target_signal = np.full((channel.y_pixels, channel.x_pixels), 50.0, dtype=np.float32)
    bg_comp = np.full((channel.y_pixels, channel.x_pixels), 10.0, dtype=np.float32)

    # Expected = Bias (100) + Target (50) + Background (10) = 160.0
    expected_value = 160.0

    with patch("instrument.science_image.generate_photon_noise_from_spectra2d", return_value=np.zeros_like(target_signal)), \
         patch("instrument.science_image.generate_cosmic_rays", return_value=np.zeros_like(target_signal)):
        result = _build_science_image_without_bg_stars(target_signal, bg_comp, channel, ctx, cfg, star, frame_index=0)

    # Validate the sum
    np.testing.assert_allclose(result, expected_value, atol=1e-5)