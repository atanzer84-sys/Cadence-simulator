"""Tests for instrument.spectrum_spread.

Sections follow the dependency chain used in production: detector placement helpers,
low-level 2D spread, the 1D→2D dispatcher, dispersion smear, then the target-star
entry point that composes placement + spread.
"""
import logging
import numpy as np
import pytest
from unittest.mock import ANY, patch

from instrument.spectrum_spread import (
    _gaussian_vertical_profile,
    _spread_1d_to_2d_gaussian,
    _spread_1d_to_2d_profile,
    get_spectrum_placement,
    get_target_star_detector_position,
    smear_1d_spectrum_dispersion,
    spread_1d_spectrum_to_2d,
    spread_target_star_spectropolarimetry_to_2d,
    spread_target_star_spectrum_to_2d,
)
# ---------------------------------------------------------------------------
# get_target_star_detector_position
# ---------------------------------------------------------------------------


# Tests: test_get_target_star_detector_position_center
# Behavior: Zero slit offsets place the target at x zero and detector center
def test_get_target_star_detector_position_center(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
    )

    x0, y0 = get_target_star_detector_position(channel)

    assert x0 == 0
    assert y0 == channel.y_pixels // 2


# Tests: test_get_target_star_detector_position_vertical_offset
# Behavior: Vertical slit offset changes the detector row
def test_get_target_star_detector_position_vertical_offset(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=2.0,
    )

    x0, y0 = get_target_star_detector_position(channel)

    assert x0 == 0
    assert y0 == (channel.y_pixels // 2) + 2


# Tests: test_get_target_star_detector_position_rejects_horizontal_shift
# Behavior: Nonzero horizontal slit offset is rejected
def test_get_target_star_detector_position_rejects_horizontal_shift(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        slit_position_x_arcsec=1.0,
        slit_position_y_arcsec=0.0,
    )

    with pytest.raises(ValueError, match="Horizontal slit_position_x_arcsec"):
        get_target_star_detector_position(channel)


# Tests: test_get_target_star_detector_position_rejects_below_detector
# Behavior: Vertical slit offset below detector is rejected
def test_get_target_star_detector_position_rejects_below_detector(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=-6.0,
    )

    with pytest.raises(ValueError, match="slit_position_y_arcsec places spectrum outside detector"):
        get_target_star_detector_position(channel)


# Tests: test_get_target_star_detector_position_rejects_above_detector
# Behavior: Vertical slit offset above detector is rejected
def test_get_target_star_detector_position_rejects_above_detector(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=5.0,
    )

    with pytest.raises(ValueError, match="slit_position_y_arcsec places spectrum outside detector"):
        get_target_star_detector_position(channel)


# ---------------------------------------------------------------------------
# get_spectrum_placement
# ---------------------------------------------------------------------------


# Tests: test_get_spectrum_placement_returns_centered_values
# Behavior: Valid channel returns target placement and zero slope/intercept
def test_get_spectrum_placement_returns_centered_values(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
        pixel_scale=1.0,
        slope=0.0,
        intercept_pixels=0.0,
    )

    x0, y0, slope, intercept = get_spectrum_placement(channel)

    assert x0 == 0
    assert y0 == float(channel.y_pixels // 2)
    assert slope == 0.0
    assert intercept == 0.0


# Tests: test_get_spectrum_placement_respects_vertical_slit_offset
# Behavior: Vertical slit offset changes returned y position
def test_get_spectrum_placement_respects_vertical_slit_offset(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=9,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=2.0,
        pixel_scale=1.0,
        slope=0.0,
        intercept_pixels=0.0,
    )

    x0, y0, slope, intercept = get_spectrum_placement(channel)

    assert x0 == 0
    assert y0 == float((channel.y_pixels // 2) + 2)
    assert slope == 0.0
    assert intercept == 0.0


# Tests: test_get_spectrum_placement_rejects_nonzero_slope
# Behavior: Nonzero slope is rejected
def test_get_spectrum_placement_rejects_nonzero_slope(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        slope=1.0,
        intercept_pixels=0.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
    )

    with pytest.raises(ValueError, match="slope and intercept_pixels must be 0"):
        get_spectrum_placement(channel)


# Tests: test_get_spectrum_placement_rejects_nonzero_intercept
# Behavior: Nonzero intercept is rejected
def test_get_spectrum_placement_rejects_nonzero_intercept(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        slope=0.0,
        intercept_pixels=1.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
    )

    with pytest.raises(ValueError, match="slope and intercept_pixels must be 0"):
        get_spectrum_placement(channel)


# Tests: test_get_spectrum_placement_propagates_horizontal_shift_error
# Behavior: Horizontal slit shift is rejected
def test_get_spectrum_placement_propagates_horizontal_shift_error(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        slit_position_x_arcsec=1.0,
        slit_position_y_arcsec=0.0,
        pixel_scale=1.0,
        slope=0.0,
        intercept_pixels=0.0,
    )

    with pytest.raises(ValueError, match="Horizontal slit_position_x_arcsec"):
        get_spectrum_placement(channel)


# Tests: test_get_spectrum_placement_propagates_vertical_out_of_bounds_error
# Behavior: Out-of-detector vertical placement is rejected
def test_get_spectrum_placement_propagates_vertical_out_of_bounds_error(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        y_pixels=10,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=-6.0,
        pixel_scale=1.0,
        slope=0.0,
        intercept_pixels=0.0,
    )

    with pytest.raises(ValueError, match="slit_position_y_arcsec places spectrum outside detector"):
        get_spectrum_placement(channel)


# ---------------------------------------------------------------------------
# _gaussian_vertical_profile, _spread_1d_to_2d_gaussian, _spread_1d_to_2d_profile
# ---------------------------------------------------------------------------


# Tests: test_gaussian_vertical_profile_normalized
# Behavior: Gaussian vertical profile is normalized and nonnegative
def test_gaussian_vertical_profile_normalized():
    profile = _gaussian_vertical_profile(ny=7, y_center=3.0, sigma=2.0)

    assert profile.shape == (7,)
    assert np.isfinite(profile).all()
    assert np.all(profile >= 0.0)
    assert np.isclose(float(profile.sum()), 1.0)


# Tests: test_gaussian_vertical_profile_peak_near_center
# Behavior: Gaussian vertical profile peaks at the requested center row
def test_gaussian_vertical_profile_peak_near_center():
    profile = _gaussian_vertical_profile(ny=9, y_center=4.0, sigma=1.5)

    assert int(np.argmax(profile)) == 4


# Tests: test_spread_1d_to_2d_gaussian_basic
# Behavior: Gaussian spread preserves column counts and returns finite float32 image
def test_spread_1d_to_2d_gaussian_basic(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        spread_half_height_pix=2,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        slope=0.0,
        intercept_pixels=0.0,
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    _spread_1d_to_2d_gaussian(image, counts, channel, placement)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert np.allclose(image.sum(axis=0), counts)


# Tests: test_spread_1d_to_2d_gaussian_rejects_missing_spread_height
# Behavior: Nonpositive spread height raises ValueError
def test_spread_1d_to_2d_gaussian_rejects_missing_spread_height(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        spread_half_height_pix=0,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        slope=0.0,
        intercept_pixels=0.0,
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with pytest.raises(ValueError, match="No cross-dispersion spreading configured"):
        _spread_1d_to_2d_gaussian(image, counts, channel, placement)


# Tests: test_spread_1d_to_2d_gaussian_preserves_column_sums
# Behavior: Gaussian spread conserves per-column counts
def test_spread_1d_to_2d_gaussian_preserves_column_sums(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        spread_half_height_pix=2,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        slope=0.0,
        intercept_pixels=0.0,
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    _spread_1d_to_2d_gaussian(image, counts, channel, placement)
    assert np.allclose(image.sum(axis=0), counts)


# Tests: test_spread_1d_to_2d_gaussian_nonzero_slope_path
# Behavior: Nonzero slope/intercept uses per-column y-center branch.
def test_spread_1d_to_2d_gaussian_nonzero_slope_path(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=11,
        spread_half_height_pix=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    _spread_1d_to_2d_gaussian(image, counts, channel, (0, 4.0, 0.5, 0.0))

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)
    peak_rows = np.argmax(image, axis=0)
    assert np.array_equal(peak_rows, np.array([4, 4, 5, 5, 6]))


# Tests: test_spread_1d_to_2d_profile_basic
# Behavior: Profile spread preserves column counts and returns finite float32 image
def test_spread_1d_to_2d_profile_basic(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_y_positions=np.array([-1, 0, 1], dtype=float),
        spread_y_weights=np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
        effective_area_wavelength=np.linspace(100, 200, 5, dtype=float),
    )
    counts = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    _spread_1d_to_2d_profile(image, counts, channel, placement)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert np.allclose(image.sum(axis=0), counts)


# Tests: test_spread_1d_to_2d_profile_detector_wavelength_mismatch
# Behavior: Detector wavelength grid length mismatch raises ValueError
def test_spread_1d_to_2d_profile_detector_wavelength_mismatch(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_y_positions=np.array([-1, 0, 1], dtype=float),
        spread_y_weights=np.ones((3, 5), dtype=np.float32),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
        effective_area_wavelength=np.linspace(100, 200, 6, dtype=float),
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with pytest.raises(ValueError, match="Detector wavelength grid length mismatch"):
        _spread_1d_to_2d_profile(image, counts, channel, placement)


# Tests: test_spread_1d_to_2d_profile_out_of_bounds_y_positions
# Behavior: Fully out-of-bounds profile rows fail conservation and log a warning
def test_spread_1d_to_2d_profile_out_of_bounds_y_positions(make_spectroscopy_channel, caplog):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_y_positions=np.array([100, 200, 300], dtype=float),
        spread_y_weights=np.ones((3, 5), dtype=np.float32),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
        effective_area_wavelength=np.linspace(100, 200, 5, dtype=float),
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        _spread_1d_to_2d_profile(image, counts, channel, placement)
    assert any("PROFILE SPREAD CHECK WARN" in r.getMessage() for r in caplog.records)


# Tests: test_spread_1d_to_2d_profile_weight_shape_mismatch
# Behavior: Weight row mismatch raises IndexError
def test_spread_1d_to_2d_profile_weight_shape_mismatch(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_y_positions=np.array([-1, 0, 1], dtype=float),
        spread_y_weights=np.ones((2, 5), dtype=np.float32),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
        effective_area_wavelength=np.linspace(100, 200, 5, dtype=float),
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)
    placement = get_spectrum_placement(channel)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with pytest.raises(IndexError):
        _spread_1d_to_2d_profile(image, counts, channel, placement)


# ---------------------------------------------------------------------------
# spread_1d_spectrum_to_2d
# ---------------------------------------------------------------------------


# Tests: test_spread_1d_spectrum_to_2d_dispatches_gaussian
# Behavior: Missing spread profile inputs dispatch to Gaussian spread
def test_spread_1d_spectrum_to_2d_dispatches_gaussian(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce"), \
         patch("instrument.spectrum_spread._spread_1d_to_2d_gaussian") as mock_gaussian, \
         patch("instrument.spectrum_spread._spread_1d_to_2d_profile") as mock_profile:
        spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0))

    assert image.shape == (7, 5)
    mock_gaussian.assert_called_once_with(image, counts, channel, (0, 3.0, 0.0, 0.0))
    mock_profile.assert_not_called()


# Tests: test_spread_1d_spectrum_to_2d_dispatches_profile
# Behavior: Present spread profile inputs dispatch to profile spread
def test_spread_1d_spectrum_to_2d_dispatches_profile(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=1,
        spread_y_positions=np.array([-1, 0, 1], dtype=float),
        spread_y_weights=np.ones((3, 5), dtype=np.float32),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce"), \
         patch("instrument.spectrum_spread._spread_1d_to_2d_profile") as mock_profile, \
         patch("instrument.spectrum_spread._spread_1d_to_2d_gaussian") as mock_gaussian:
        spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0))

    assert image.shape == (7, 5)
    mock_profile.assert_called_once_with(image, counts, channel, (0, 3.0, 0.0, 0.0))
    mock_gaussian.assert_not_called()


# Tests: test_spread_1d_spectrum_to_2d_counts_length_mismatch
# Behavior: Counts length mismatch raises ValueError
def test_spread_1d_spectrum_to_2d_counts_length_mismatch(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.array([1, 2, 3], dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce"):
        with pytest.raises(ValueError, match="Counts length 3 does not match nx 5"):
            spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0))


# Tests: test_spread_1d_spectrum_to_2d_mode_not_implemented
# Behavior: Unsupported mode raises NotImplementedError
def test_spread_1d_spectrum_to_2d_mode_not_implemented(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=2,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.ones(5, dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce"):
        with pytest.raises(NotImplementedError, match="mode=2 not implemented"):
            spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0))


# Tests: test_spread_1d_spectrum_to_2d_announces_by_default
# Behavior: Default call announces spreading to the user
def test_spread_1d_spectrum_to_2d_announces_by_default(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.ones(5, dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce") as mock_announce, \
         patch("instrument.spectrum_spread._spread_1d_to_2d_gaussian"):
        spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0))

    mock_announce.assert_called_once_with(
        f"Spreading 1D counts to 2D detector image for channel {channel.channel_name}.",
        to_user=True,
    )


# Tests: test_spread_1d_spectrum_to_2d_suppresses_user_announce
# Behavior: announce_user False suppresses user-facing announce flag
def test_spread_1d_spectrum_to_2d_suppresses_user_announce(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        mode=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.ones(5, dtype=np.float32)

    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
    with patch("instrument.spectrum_spread.announce") as mock_announce, \
         patch("instrument.spectrum_spread._spread_1d_to_2d_gaussian"):
        spread_1d_spectrum_to_2d(image, counts, channel, (0, 3.0, 0.0, 0.0), announce_user=False)

    mock_announce.assert_called_once_with(
        f"Spreading 1D counts to 2D detector image for channel {channel.channel_name}.",
        to_user=False,
    )


# ---------------------------------------------------------------------------
# smear_1d_spectrum_dispersion
# ---------------------------------------------------------------------------


# Tests: test_smear_1d_spectrum_dispersion_no_smear
# Behavior: Smear shift of zero returns unchanged counts
def test_smear_1d_spectrum_dispersion_no_smear(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(smear_shift_pixels=0.0)
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    smeared = smear_1d_spectrum_dispersion(counts, channel)

    assert smeared.shape == counts.shape
    assert np.array_equal(smeared, counts)
    assert smeared is not counts


# Tests: test_smear_1d_spectrum_dispersion_single_step
# Behavior: Smear shift of one returns unchanged counts
def test_smear_1d_spectrum_dispersion_single_step(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(smear_shift_pixels=1.0)
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    smeared = smear_1d_spectrum_dispersion(counts, channel)

    assert smeared.shape == counts.shape
    assert np.array_equal(smeared, counts)
    assert smeared is not counts


# Tests: test_smear_1d_spectrum_dispersion_preserves_total_counts
# Behavior: Multi-step smear preserves total counts
def test_smear_1d_spectrum_dispersion_preserves_total_counts(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(smear_shift_pixels=3.0)
    counts = np.array([0, 0, 9, 0, 0], dtype=np.float32)

    smeared = smear_1d_spectrum_dispersion(counts, channel)

    assert smeared.shape == counts.shape
    assert np.isfinite(smeared).all()
    assert np.isclose(float(smeared.sum()), float(counts.sum()))
    assert not np.array_equal(smeared, counts)


# Tests: test_smear_1d_spectrum_dispersion_even_steps
# Behavior: Even smear shift spreads counts and preserves total
def test_smear_1d_spectrum_dispersion_even_steps(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(smear_shift_pixels=4.0)
    counts = np.array([0, 0, 8, 0, 0], dtype=np.float32)

    smeared = smear_1d_spectrum_dispersion(counts, channel)

    assert smeared.shape == counts.shape
    assert np.isclose(float(smeared.sum()), float(counts.sum()))
    assert np.all(smeared >= 0.0)
    assert not np.array_equal(smeared, counts)


# ---------------------------------------------------------------------------
# spread_target_star_spectrum_to_2d
# ---------------------------------------------------------------------------


# Tests: test_spread_target_star_spectrum_to_2d_gaussian
# Behavior: Target-star Gaussian spread preserves column counts
def test_spread_target_star_spectrum_to_2d_gaussian(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_half_height_pix=2,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    with patch("instrument.spectrum_spread.announce"):
        image = spread_target_star_spectrum_to_2d(counts, channel)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert np.allclose(image.sum(axis=0), counts)


# Tests: test_spread_target_star_spectrum_to_2d_profile
# Behavior: Target-star profile spread preserves column counts
def test_spread_target_star_spectrum_to_2d_profile(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        spread_half_height_pix=2,
        spread_y_positions=np.array([-1, 0, 1], dtype=float),
        spread_y_weights=np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.6, 0.6, 0.6, 0.6, 0.6],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        spread_y_wavelengths=np.linspace(100, 200, 5, dtype=float),
        effective_area_wavelength=np.linspace(100, 200, 5, dtype=float),
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
    )
    counts = np.array([10, 20, 30, 40, 50], dtype=np.float32)

    with patch("instrument.spectrum_spread.announce"):
        image = spread_target_star_spectrum_to_2d(counts, channel)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert np.allclose(image.sum(axis=0), counts)


# Tests: test_spread_target_star_spectrum_to_2d_rejects_nonzero_slope
# Behavior: Nonzero slope is rejected before spreading
def test_spread_target_star_spectrum_to_2d_rejects_nonzero_slope(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=1.0,
        intercept_pixels=0.0,
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)

    with pytest.raises(ValueError, match="slope and intercept_pixels must be 0"):
        spread_target_star_spectrum_to_2d(counts, channel)


# Tests: test_spread_target_star_spectrum_to_2d_rejects_nonzero_intercept
# Behavior: Nonzero intercept is rejected before spreading
def test_spread_target_star_spectrum_to_2d_rejects_nonzero_intercept(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=1.0,
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)

    with pytest.raises(ValueError, match="slope and intercept_pixels must be 0"):
        spread_target_star_spectrum_to_2d(counts, channel)


# Tests: test_spread_target_star_spectrum_to_2d_counts_length_mismatch
# Behavior: Counts length mismatch raises from spreading step
def test_spread_target_star_spectrum_to_2d_counts_length_mismatch(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
    )
    counts = np.array([1, 2, 3], dtype=np.float32)

    with patch("instrument.spectrum_spread.announce"):
        with pytest.raises(ValueError, match="Counts length 3 does not match nx 5"):
            spread_target_star_spectrum_to_2d(counts, channel)


# Tests: test_spread_target_star_spectrum_to_2d_mode_not_implemented
# Behavior: Unsupported mode raises from spreading step
def test_spread_target_star_spectrum_to_2d_mode_not_implemented(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=7,
        slope=0.0,
        intercept_pixels=0.0,
        mode=2,
    )
    counts = np.ones(channel.x_pixels, dtype=np.float32)

    with patch("instrument.spectrum_spread.announce"):
        with pytest.raises(NotImplementedError, match="mode=2 not implemented"):
            spread_target_star_spectrum_to_2d(counts, channel)


# Tests: test_spread_target_star_spectrum_to_2d_respects_vertical_slit_offset
# Behavior: Vertical slit offset shifts the target spectrum position
def test_spread_target_star_spectrum_to_2d_respects_vertical_slit_offset(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=9,
        slope=0.0,
        intercept_pixels=0.0,
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=2.0,
        spread_half_height_pix=1,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
    )
    counts = np.zeros(channel.x_pixels, dtype=np.float32)
    counts[0] = 1.0

    with patch("instrument.spectrum_spread.announce"):
        image = spread_target_star_spectrum_to_2d(counts, channel)

    peak_y = int(np.argmax(image[:, 0]))
    assert peak_y != channel.y_pixels // 2



# Tests: test_spread_target_star_spectrum_to_2d_dispatches_spectropolarimetry
# Behavior: Spectropolarimetry mode dispatches to the dedicated helper
def test_spread_target_star_spectrum_to_2d_dispatches_spectropolarimetry(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        observation_mode="spectropolarimetry",
        beam_separation_pix=2,
        polarization_wavelength=np.array([200, 250, 300, 350, 400], dtype=float),
        polarization_delta=np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float32),
    )
    counts = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    with patch("instrument.spectrum_spread.get_spectrum_placement", return_value=(0, 3.0, 0.0, 0.0)) as mock_placement, \
         patch("instrument.spectrum_spread.spread_target_star_spectropolarimetry_to_2d", return_value=np.ones((channel.y_pixels, channel.x_pixels), dtype=np.float32)) as mock_pol:
        image = spread_target_star_spectrum_to_2d(counts, channel)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    mock_placement.assert_called_once_with(channel)
    mock_pol.assert_called_once_with(ANY, counts, channel, (0, 3.0, 0.0, 0.0))


# Tests: test_spread_target_star_spectropolarimetry_to_2d_splits_flux_and_separates_beam
# Behavior: Spectropolarimetry splits flux by delta, shifts beam 2 vertically, and preserves total flux
def test_spread_target_star_spectropolarimetry_to_2d_splits_flux_and_separates_beam(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=6,
        observation_mode="spectropolarimetry",
        beam_separation_pix=2,
        effective_area_wavelength=np.array([200, 250, 300, 350, 400], dtype=float),
        polarization_wavelength=np.array([200, 250, 300, 350, 400], dtype=float),
        polarization_delta=np.array([1.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32),
    )
    counts = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    placement = (0, 3.0, 0.0, 0.0)

    def _fake_spread(image, counts_1d, _channel, _placement, announce_user=True):
        image[0, :] = counts_1d

    with patch("instrument.spectrum_spread.spread_1d_spectrum_to_2d", side_effect=_fake_spread) as mock_spread:
        image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
        spread_target_star_spectropolarimetry_to_2d(image, counts, channel, placement)

    expected_delta = np.array([1.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    expected_beam1 = counts * (1.0 + expected_delta) / 2.0
    expected_beam2 = counts * (1.0 - expected_delta) / 2.0

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert image.dtype == np.float32
    assert np.allclose(image[0, :], expected_beam1)
    assert np.allclose(image[1, :], 0.0)
    assert np.allclose(image[2, :], expected_beam2)
    assert np.allclose(image.sum(), counts.sum())

    assert mock_spread.call_count == 2
    assert np.allclose(mock_spread.call_args_list[0].args[1], expected_beam1)
    assert np.allclose(mock_spread.call_args_list[1].args[1], expected_beam2)
    assert mock_spread.call_args_list[0].kwargs["announce_user"] is True
    assert mock_spread.call_args_list[1].kwargs["announce_user"] is False


# Tests: test_spread_target_star_spectropolarimetry_to_2d_interpolates_delta
# Behavior: Polarization delta is interpolated onto the detector wavelength grid before beam split
def test_spread_target_star_spectropolarimetry_to_2d_interpolates_delta(make_spectroscopy_channel):
    channel = make_spectroscopy_channel(
        x_pixels=5,
        y_pixels=6,
        observation_mode="spectropolarimetry",
        beam_separation_pix=1,
        effective_area_wavelength=np.array([200, 225, 250, 275, 300], dtype=float),
        polarization_wavelength=np.array([200, 250, 300], dtype=float),
        polarization_delta=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    counts = np.array([10, 10, 10, 10, 10], dtype=np.float32)
    placement = (0, 3.0, 0.0, 0.0)

    def _fake_spread(image, counts_1d, _channel, _placement, announce_user=True):
        image[0, :] = counts_1d

    with patch("instrument.spectrum_spread.spread_1d_spectrum_to_2d", side_effect=_fake_spread) as mock_spread:
        image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)
        spread_target_star_spectropolarimetry_to_2d(image, counts, channel, placement)

    expected_delta = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float32)
    expected_beam1 = counts * (1.0 + expected_delta) / 2.0
    expected_beam2 = counts * (1.0 - expected_delta) / 2.0

    assert np.allclose(mock_spread.call_args_list[0].args[1], expected_beam1)
    assert np.allclose(mock_spread.call_args_list[1].args[1], expected_beam2)
    assert np.allclose(image[0, :], expected_beam1)
    assert np.allclose(image[1, :], expected_beam2)
    assert np.allclose(image.sum(), counts.sum())