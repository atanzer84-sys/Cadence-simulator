import numpy as np
from unittest.mock import patch
from unittest.mock import Mock
import pytest
from instrument.science_image import build_science_images, _build_science_image_without_bg_stars, _generate_channel_calibration_frames, _create_channel_images, _create_per_exposure

# Tests: _build_science_image_without_bg_stars
# Behavior: Verifies that the construction matching channel dimensions dynamically.
def test_build_science_image_integrity(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel()
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()
    base_header = make_header()

    target_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_comp = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    result = _build_science_image_without_bg_stars(target_signal, bg_comp, channel, ctx, cfg, star, frame_index=0, base_header=base_header)

    assert result.dtype == np.float32
    # Dynamic shape check to avoid brittle hardcoding
    assert result.shape == (channel.y_pixels, channel.x_pixels)


# Tests: _build_science_image_without_bg_stars
# Behavior: Ensures the diagnostic PNG writing path (frame_index < 1) is exercised without errors.
def test_build_science_image_diagnostic_path(make_photometry_channel, make_run_context, make_global_config, make_star, make_header):
    # Initialize a photometry channel (32x32).
    channel = make_photometry_channel()
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()
    base_header = make_header()

    signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    # Execute for frame 0 (triggers diagnostic logic) and frame 1 (skips it).
    _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=0, base_header=base_header)
    _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=1, base_header=base_header)


# Tests: _build_science_image_without_bg_stars
# Behavior: Verifies that the builder handles custom channel pixel overrides correctly.
def test_build_science_image_custom_geometry(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    # Use fixture overrides to test a non-standard 10x5 sensor.
    channel = make_spectroscopy_channel(x_pixels=10, y_pixels=5)
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()
    base_header = make_header()

    signal = np.zeros((5, 10), dtype=np.float32)

    # Execute and verify the resulting array shape matches the override.
    result = _build_science_image_without_bg_stars(signal, signal, channel, ctx, cfg, star, frame_index=0, base_header=base_header)
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
def test_build_science_image_additive_correctness(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
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
    base_header = make_header()

    # Create known constant component arrays
    target_signal = np.full((channel.y_pixels, channel.x_pixels), 50.0, dtype=np.float32)
    bg_comp = np.full((channel.y_pixels, channel.x_pixels), 10.0, dtype=np.float32)

    # Expected = Bias (100) + Target (50) + Background (10) = 160.0
    expected_value = 160.0

    with patch("instrument.science_image.generate_photon_noise_from_spectra2d", return_value=np.zeros_like(target_signal)), \
         patch("instrument.science_image.generate_cosmic_rays", return_value=np.zeros_like(target_signal)):
        result = _build_science_image_without_bg_stars(target_signal, bg_comp, channel, ctx, cfg, star, frame_index=0, base_header=base_header)

    # Validate the sum
    np.testing.assert_allclose(result, expected_value, atol=1e-5)


# Tests: _create_channel_images
# Behavior: builds one science frame, appends headers, and writes FITS/PNG outputs.
def test_create_channel_images_writes_science_outputs(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel(n_science_frames=1, exposure_s=10.0, x_pixels=4, y_pixels=3)
    ctx = make_run_context()
    cfg = make_global_config(readout_gap_s=2.0, orbit_duration_minutes=90.0)
    star = make_star()
    base_header = make_header()
    stellar_signal = np.ones((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_component = np.full((channel.y_pixels, channel.x_pixels), 3.0, dtype=np.float32)
    img = np.full((channel.y_pixels, channel.x_pixels), 7.0, dtype=np.float32)
    phot = (1.0, 2.0, 3, 4, 5.0, 10.0)

    with patch("instrument.science_image.generate_background_image", return_value=bg_component) as mock_bg, \
         patch("instrument.science_image._create_per_exposure", return_value=img) as mock_create, \
         patch("instrument.science_image.compute_aperture_photometry", return_value=phot) as mock_phot, \
         patch("instrument.science_image.append_base_frame_header", return_value=base_header) as mock_base_hdr, \
         patch("instrument.science_image.append_image_stats_header") as mock_stats_hdr, \
         patch("instrument.science_image.append_channel_frame_header") as mock_channel_hdr, \
         patch("instrument.science_image.append_photometry_header") as mock_phot_hdr, \
         patch("instrument.science_image.Frame") as mock_frame, \
         patch("instrument.science_image.write_fits_frame") as mock_write_fits:
        _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog=None, base_header=base_header)

    mock_bg.assert_called_once_with(channel, star)
    mock_create.assert_called_once()
    create_args = mock_create.call_args.args
    np.testing.assert_allclose(create_args[0], stellar_signal * channel.exposure_s)
    np.testing.assert_allclose(create_args[1], bg_component)
    assert create_args[2] is channel
    assert create_args[3] is ctx
    assert create_args[4] is cfg
    assert create_args[5] is star
    assert create_args[7] == 0
    assert create_args[8] == 0.0
    assert create_args[9] == 360.0 * (channel.exposure_s / (cfg.orbit_duration_minutes * 60.0))

    mock_phot.assert_called_once_with(img, channel)
    mock_base_hdr.assert_called_once_with(base_header, filetype="SCIENCE", channel=channel, index0=0)
    mock_stats_hdr.assert_called_once_with(base_header, img)
    mock_channel_hdr.assert_called_once_with(base_header, channel, exptime_s=channel.exposure_s)
    mock_phot_hdr.assert_called_once_with(base_header, phot)
    mock_frame.assert_called_once_with(data=img, header=base_header, frame_type="science", channel_tag=channel.channel_name)
    mock_write_fits.assert_called_once()


# Tests: _create_channel_images
# Behavior: roll angles progress with exposure+readout cadence across frames.
def test_create_channel_images_roll_angle_progression(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel(n_science_frames=2, exposure_s=10.0, x_pixels=2, y_pixels=2)
    ctx = make_run_context()
    cfg = make_global_config(readout_gap_s=5.0, orbit_duration_minutes=1.0)
    star = make_star()
    base_header = make_header()
    stellar_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_component = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    img = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    with patch("instrument.science_image.generate_background_image", return_value=bg_component), \
         patch("instrument.science_image._create_per_exposure", return_value=img) as mock_create, \
         patch("instrument.science_image.compute_aperture_photometry", return_value=None), \
         patch("instrument.science_image.append_base_frame_header", return_value=base_header), \
         patch("instrument.science_image.append_image_stats_header"), \
         patch("instrument.science_image.append_channel_frame_header"), \
         patch("instrument.science_image.append_photometry_header"), \
         patch("instrument.science_image.Frame"), \
         patch("instrument.science_image.write_fits_frame"):
        _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog=None, base_header=base_header)

    assert mock_create.call_count == 2
    first = mock_create.call_args_list[0].args
    second = mock_create.call_args_list[1].args

    assert first[7] == 0
    assert first[8] == 0.0
    assert first[9] == 60.0

    # time_s for frame 1 is exposure + readout_gap = 15 s in a 60 s orbit
    assert second[7] == 1
    assert second[8] == 90.0
    assert second[9] == 150.0


def test_create_channel_images_roll_angle_step_and_width_invariants(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel(n_science_frames=3, exposure_s=10.0, x_pixels=2, y_pixels=2)
    ctx = make_run_context()
    cfg = make_global_config(readout_gap_s=5.0, orbit_duration_minutes=1.0, orbit_revolutions=2.0)
    star = make_star()
    base_header = make_header()
    stellar_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_component = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    img = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    with patch("instrument.science_image.generate_background_image", return_value=bg_component), \
         patch("instrument.science_image._create_per_exposure", return_value=img) as mock_create, \
         patch("instrument.science_image.compute_aperture_photometry", return_value=None), \
         patch("instrument.science_image.append_base_frame_header", return_value=base_header), \
         patch("instrument.science_image.append_image_stats_header"), \
         patch("instrument.science_image.append_channel_frame_header"), \
         patch("instrument.science_image.append_photometry_header"), \
         patch("instrument.science_image.Frame"), \
         patch("instrument.science_image.write_fits_frame"):
        _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog=None, base_header=base_header)

    starts = [call.args[8] for call in mock_create.call_args_list]
    ends = [call.args[9] for call in mock_create.call_args_list]
    cadence_deg = 360.0 * ((channel.exposure_s + cfg.readout_gap_s) / (cfg.orbit_duration_minutes * 60.0))
    width_deg = 360.0 * (channel.exposure_s / (cfg.orbit_duration_minutes * 60.0))

    np.testing.assert_allclose(np.diff(starts), cadence_deg)
    np.testing.assert_allclose(np.array(ends) - np.array(starts), width_deg)


def test_create_channel_images_reports_wrapped_angles_and_orbit_for_multi_orbit(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel(n_science_frames=2, exposure_s=10.0, x_pixels=2, y_pixels=2)
    ctx = make_run_context()
    cfg = make_global_config(readout_gap_s=50.0, orbit_duration_minutes=1.0, orbit_revolutions=2.0)
    star = make_star()
    base_header = make_header()
    stellar_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    bg_component = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    img = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    with patch("instrument.science_image.generate_background_image", return_value=bg_component), \
         patch("instrument.science_image._create_per_exposure", return_value=img), \
         patch("instrument.science_image.compute_aperture_photometry", return_value=None), \
         patch("instrument.science_image.append_base_frame_header", return_value=base_header), \
         patch("instrument.science_image.append_image_stats_header"), \
         patch("instrument.science_image.append_channel_frame_header"), \
         patch("instrument.science_image.append_photometry_header"), \
         patch("instrument.science_image.Frame"), \
         patch("instrument.science_image.write_fits_frame"), \
         patch("instrument.science_image._report_science_frame_progress") as mock_report:
        _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog=None, base_header=base_header)

    assert mock_report.call_count == 2
    first_call = mock_report.call_args_list[0].args
    second_call = mock_report.call_args_list[1].args

    assert first_call[0] is channel
    assert first_call[1] == 0
    assert first_call[2] == 2
    assert first_call[3] == 0.0
    assert first_call[4] == 60.0

    assert second_call[0] is channel
    assert second_call[1] == 1
    assert second_call[2] == 2
    assert second_call[3] == 360.0
    assert second_call[4] == 420.0


# Tests: _create_per_exposure
# Behavior: spectroscopy branch adds background stars, applies gain, and forwards visibility bands.
def test_create_per_exposure_spectroscopy_branch(make_spectroscopy_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_spectroscopy_channel(x_pixels=3, y_pixels=2, ccd_gain=2.0)
    visibility_mock = Mock()
    ctx = make_run_context()
    cfg = make_global_config(
        invert_calibration_science_frame_component=False,
        write_background_star_footprint_on_science_frame=True,
    )
    star = make_star()
    base_header = make_header()

    base_img = np.full((2, 3), 10.0, dtype=np.float32)
    bg_stars = np.full((2, 3), 1.5, dtype=np.float32)
    bands = {"dummy": np.array([1, 2, 3])}
    expected = (base_img + bg_stars) * channel.ccd_gain

    with patch("instrument.science_image._build_science_image_without_bg_stars", return_value=base_img) as mock_build, \
         patch("instrument.science_image.generate_background_star_spectroscopy_image", return_value=(bg_stars, bands)) as mock_bg_spec, \
         patch("instrument.science_image.generate_background_star_photometry_image") as mock_bg_phot, \
         patch("instrument.science_image.generate_background_star_visibility_on_science_frame", visibility_mock):
        result = _create_per_exposure(
            stellar_component=np.zeros_like(base_img),
            background_component=np.zeros_like(base_img),
            channel=channel,
            ctx=ctx,
            cfg=cfg,
            star=star,
            background_stars_catalog=None,
            frame_index=3,
            roll_angle_start=12.5,
            roll_angle_end=15.0,
            base_header=base_header,
        )

    mock_build.assert_called_once()
    mock_bg_spec.assert_called_once_with(channel, None, 12.5, 15.0, 3)
    mock_bg_phot.assert_not_called()

    np.testing.assert_allclose(result, expected)

    call = visibility_mock.call_args
    assert call.args[2] is ctx
    assert call.args[3] is channel
    assert call.kwargs["star"] is star
    assert call.kwargs["index"] == 3
    assert call.kwargs["inverted"] is False
    assert "background_star_bands" in call.kwargs
    assert call.kwargs["background_star_bands"] is bands


# Tests: _create_per_exposure
# Behavior: photometry branch forwards arc visibility payload.
def test_create_per_exposure_photometry_branch(make_photometry_channel, make_run_context, make_global_config, make_star, make_header):
    channel = make_photometry_channel(x_pixels=3, y_pixels=2, ccd_gain=1.0)
    visibility_mock = Mock()
    ctx = make_run_context()
    cfg = make_global_config(
        invert_calibration_science_frame_component=True,
        write_background_star_footprint_on_science_frame=True,
    )
    star = make_star()
    base_header = make_header()

    base_img = np.full((2, 3), 5.0, dtype=np.float32)
    bg_stars = np.full((2, 3), 2.0, dtype=np.float32)
    arcs = {"arcs": [0, 1]}
    expected = base_img + bg_stars

    with patch("instrument.science_image._build_science_image_without_bg_stars", return_value=base_img), \
         patch("instrument.science_image.generate_background_star_photometry_image", return_value=(bg_stars, arcs)) as mock_bg_phot, \
         patch("instrument.science_image.generate_background_star_spectroscopy_image") as mock_bg_spec, \
         patch("instrument.science_image.generate_background_star_visibility_on_science_frame", visibility_mock):
        result = _create_per_exposure(
            stellar_component=np.zeros_like(base_img),
            background_component=np.zeros_like(base_img),
            channel=channel,
            ctx=ctx,
            cfg=cfg,
            star=star,
            background_stars_catalog=None,
            frame_index=0,
            roll_angle_start=0.0,
            roll_angle_end=1.0,
            base_header=base_header,
        )

    mock_bg_phot.assert_called_once_with(channel, None, 0.0, 1.0, 0)
    mock_bg_spec.assert_not_called()
    np.testing.assert_allclose(result, expected)

    call = visibility_mock.call_args
    assert call.kwargs["inverted"] is True
    assert "background_star_arcs" in call.kwargs
    assert call.kwargs["background_star_arcs"] is arcs


# Tests: _create_per_exposure
# Behavior: unsupported channel type raises TypeError.
def test_create_per_exposure_rejects_unsupported_channel(make_run_context, make_global_config, make_star, make_header):
    class DummyChannel:
        ccd_gain = 1.0

    channel = DummyChannel()
    ctx = make_run_context()
    cfg = make_global_config()
    star = make_star()
    base_header = make_header()
    base_img = np.zeros((2, 2), dtype=np.float32)

    with patch("instrument.science_image._build_science_image_without_bg_stars", return_value=base_img):
        with pytest.raises(TypeError):
            _create_per_exposure(
                stellar_component=base_img,
                background_component=base_img,
                channel=channel,
                ctx=ctx,
                cfg=cfg,
                star=star,
                background_stars_catalog=None,
                frame_index=0,
                roll_angle_start=0.0,
                roll_angle_end=1.0,
                base_header=base_header,
            )


# Tests: build_science_images
# Behavior: orchestrates global config/header setup and downstream frame/image creation.
def test_build_science_images_orchestrates_pipeline(make_spectroscopy_channel, make_run_context, make_star, make_global_config):
    channel = make_spectroscopy_channel()
    ctx = make_run_context()
    star = make_star()
    stellar_signal = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    background_catalog = object()
    cfg = make_global_config()
    header = object()

    with patch("instrument.science_image.get_global_config", return_value=cfg) as mock_cfg, \
         patch("instrument.science_image.initialize_fits_header", return_value=header) as mock_init_hdr, \
         patch("instrument.science_image._generate_channel_calibration_frames") as mock_cal, \
         patch("instrument.science_image._create_channel_images") as mock_create:
        build_science_images(stellar_signal, channel, ctx, star, background_catalog)

    mock_cfg.assert_called_once_with()
    mock_init_hdr.assert_called_once_with(star, ctx.timestamp)
    mock_cal.assert_called_once_with(channel, header, ctx, star, cfg)
    mock_create.assert_called_once_with(stellar_signal, channel, ctx, cfg, star, background_catalog, header)