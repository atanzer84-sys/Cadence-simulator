import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from instrument.spectrum_spread import (
    spread_1d_spectrum_to_2d,
    spread_target_star_spectrum_to_2d,
    _spread_1d_to_2d_gaussian,
    _spread_1d_to_2d_profile,
    get_spectrum_placement,
    get_target_star_detector_position,
)


def _channel_gaussian():
    """Channel for Gaussian spread: spread_y_* are None."""
    c = _channel_base()
    c.spread_y_positions = None
    c.spread_y_weights = None
    c.spread_y_wavelengths = None
    return c


def _channel_profile():
    """Channel for profile spread: has spread_y_* and wavelength."""
    c = _channel_base()
    c.wavelength = np.linspace(100, 200, 5)
    c.spread_y_positions = np.array([-1, 0, 1])
    c.spread_y_weights = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.6, 0.6, 0.6, 0.6, 0.6],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ])
    c.spread_y_wavelengths = np.linspace(100, 200, 5)
    return c


def _channel_base():
    """Base channel attributes; spread mode depends on spread_y_*."""
    c = SimpleNamespace()
    c.channel_name = "NUV"
    c.x_pixels = 5
    c.y_pixels = 7
    c.pixel_scale = 1.0
    c.slit_position_x_arcsec = 0.0
    c.slit_position_y_arcsec = 0.0
    c.slope = 0.0
    c.intercept_pixels = 0.0
    c.mode = 1
    c.spread_half_height_pix = 2
    c.ccd_gain = 2.0
    c.bias_offset = 10.0
    c.read_noise = 3.0
    c.dark_current_sigma = 0.5
    c.dark_noise = 1.0
    c.exposure_s = 3.0
    c.spread_profile_file = "dummy_profile.fits"
    c.effective_area_wavelength = np.linspace(100, 200, 5)
    return c


# ----------------------------------------------------------------------
# TESTS FOR spread_1d_spectrum_to_2d (mode dispatch, counts mismatch)
# ----------------------------------------------------------------------

def test_spread_1d_spectrum_to_2d_dispatches_gaussian():
    """spread_1d_spectrum_to_2d uses Gaussian when spread_y_* are None."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    image = spread_target_star_spectrum_to_2d(counts, channel)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_spread_1d_spectrum_to_2d_dispatches_profile():
    """spread_1d_spectrum_to_2d uses profile when spread_y_* are set."""
    channel = _channel_profile()
    counts = np.array([10, 20, 30, 40, 50], dtype=float)

    image = spread_target_star_spectrum_to_2d(counts, channel)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_spread_1d_spectrum_to_2d_counts_length_mismatch():
    """spread_1d_spectrum_to_2d raises ValueError when len(counts) != nx."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3], dtype=float)  # len=3, nx=5

    with pytest.raises(ValueError, match="Counts length 3 does not match nx 5"):
        spread_target_star_spectrum_to_2d(counts, channel)


def test_spread_1d_spectrum_to_2d_mode_not_implemented():
    """spread_1d_spectrum_to_2d raises NotImplementedError when mode != 1."""
    channel = _channel_gaussian()
    channel.mode = 2
    counts = np.ones(channel.x_pixels)

    with pytest.raises(NotImplementedError, match="mode=2 not implemented"):
        spread_target_star_spectrum_to_2d(counts, channel)


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_gaussian
# ----------------------------------------------------------------------

def test_gaussian_spread_basic():
    """Gaussian spread produces correct shape and column sums match input counts."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    image = _spread_1d_to_2d_gaussian(counts, channel, x0, y0, slope, intercept)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_gaussian_spread_no_spread_height():
    """spread_half_height_pix <= 0 raises ValueError."""
    channel = _channel_gaussian()
    channel.spread_half_height_pix = 0
    counts = np.ones(channel.x_pixels)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_gaussian(counts, channel, x0, y0, slope, intercept)


def test_gaussian_spread_column_sum_mismatch_raises():
    """When column sums fail validation, _spread_1d_to_2d_gaussian raises ValueError."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    with patch("instrument.spectrum_spread.np.allclose", return_value=False):
        with pytest.raises(ValueError, match="Gaussian spread column sum mismatch"):
            _spread_1d_to_2d_gaussian(counts, channel, x0, y0, slope, intercept)


def test_gaussian_spread_with_nonzero_slope():
    """_spread_1d_to_2d_gaussian supports slope/intercept via slow path; column sums match."""
    channel = _channel_gaussian()
    channel.slope = 0.5
    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    image = _spread_1d_to_2d_gaussian(counts, channel, x0, y0, slope, intercept)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_profile
# ----------------------------------------------------------------------

def test_profile_spread_basic():
    """Profile spread produces correct shape and column sums match input counts."""
    channel = _channel_profile()
    counts = np.array([10, 20, 30, 40, 50], dtype=float)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    image = _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_profile_spread_detector_wavelength_mismatch():
    """Detector wavelength length mismatch raises ValueError."""
    channel = _channel_profile()
    channel.effective_area_wavelength = np.linspace(100, 200, 6)  # wrong length
    counts = np.ones(channel.x_pixels)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)


def test_profile_spread_out_of_bounds_y_positions():
    """Out-of-bounds y positions are skipped; no crash."""
    channel = _channel_profile()
    channel.spread_y_positions = np.array([100, 200, 300])
    counts = np.ones(channel.x_pixels)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    image = _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)

    assert image.shape == (channel.y_pixels, channel.x_pixels)


def test_profile_spread_weight_shape_mismatch():
    """Weight shape mismatch (fewer rows than spread positions) raises IndexError."""
    channel = _channel_profile()
    channel.spread_y_weights = np.ones((2, 5))  # 2 rows, 3 spread positions
    counts = np.ones(channel.x_pixels)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    with pytest.raises(IndexError):
        _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)


def test_profile_spread_rejects_nonzero_slope_or_intercept():
    """_spread_1d_to_2d_profile raises when slope or intercept_pixels is non-zero."""
    channel = _channel_profile()
    channel.slope = 1.0
    counts = np.ones(channel.x_pixels)
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    with pytest.raises(ValueError, match="slope and intercept_pixels not supported yet"):
        _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)


def test_profile_spread_respects_vertical_slit_offset():
    """When slit_position_y_arcsec is non-zero, the vertical peak is not at ny//2."""
    channel = _channel_profile()
    # ensure a reasonable detector size (center row = 4)
    channel.y_pixels = 9
    channel.pixel_scale = 1.0
    channel.slit_position_y_arcsec = 2.0  # shift anchor upward by 2 pixels

    counts = np.zeros(channel.x_pixels, dtype=float)
    counts[0] = 1.0  # only first column has flux
    x0, y0, slope, intercept = get_spectrum_placement(channel)

    image = _spread_1d_to_2d_profile(counts, channel, x0, y0, slope, intercept)

    col0 = image[:, 0]
    peak_y = int(np.argmax(col0))

    # Without offset, peak would be at ny//2; with offset it must move.
    assert peak_y != channel.y_pixels // 2



# ----------------------------------------------------------------------
# TESTS FOR get_target_star_detector_position (anchor logic)
# ----------------------------------------------------------------------


def _anchor_channel(
    x_pixels=10,
    y_pixels=10,
    pixel_scale=1.0,
    slit_x_arcsec=0.0,
    slit_y_arcsec=0.0,
):
    """Minimal channel namespace for get_target_star_detector_position tests."""
    return SimpleNamespace(
        channel_name="TEST",
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        pixel_scale=pixel_scale,
        slit_position_x_arcsec=slit_x_arcsec,
        slit_position_y_arcsec=slit_y_arcsec,
    )


def testget_target_star_detector_position_center():
    """With zero slit offsets, anchor is at detector center in y and x=0."""
    ch = _anchor_channel(y_pixels=10, slit_x_arcsec=0.0, slit_y_arcsec=0.0, pixel_scale=1.0)

    x0, y0 = get_target_star_detector_position(ch)

    assert x0 == 0
    assert y0 == ch.y_pixels // 2


def testget_target_star_detector_position_horizontal_shift_not_supported():
    """Non-zero slit_position_x_arcsec raises a ValueError."""
    ch = _anchor_channel(slit_x_arcsec=1.0, slit_y_arcsec=0.0, pixel_scale=1.0)

    with pytest.raises(ValueError, match="Horizontal slit_position_x_arcsec"):
        get_target_star_detector_position(ch)


def testget_target_star_detector_position_y_below_detector_raises():
    """Negative y0 (slit below detector) raises a ValueError."""
    # For ny=10, ny//2=5; choose slit_y_arcsec so y0 ≈ -1 → 5 + slit_y = -1 ⇒ slit_y=-6
    ch = _anchor_channel(y_pixels=10, slit_y_arcsec=-6.0, pixel_scale=1.0)

    with pytest.raises(ValueError, match="slit_position_y_arcsec places spectrum outside detector"):
        get_target_star_detector_position(ch)


def testget_target_star_detector_position_y_above_detector_raises():
    """y0 >= ny (slit above detector) raises a ValueError."""
    # For ny=10, ny//2=5; choose slit_y_arcsec so y0 ≈ 10 → 5 + slit_y = 10 ⇒ slit_y=5
    ch = _anchor_channel(y_pixels=10, slit_y_arcsec=5.0, pixel_scale=1.0)

    with pytest.raises(ValueError, match="slit_position_y_arcsec places spectrum outside detector"):
        get_target_star_detector_position(ch)
