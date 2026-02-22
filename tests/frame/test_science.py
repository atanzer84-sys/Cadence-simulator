import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from frame.science import (
    generate_science_frames,
    spread_1d_spectrum_to_2d,
    _spread_1d_to_2d_gaussian,
    _spread_1d_to_2d_profile,
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
    c.mode = 1
    c.spread_half_height_pix = 2
    c.ccd_gain = 2.0
    c.bias_offset = 10.0
    c.read_noise = 3.0
    c.dark_current_sigma = 0.5
    c.dark_noise = 1.0
    c.exposure_s = 3.0
    c.spread_profile_file = "dummy_profile.fits"
    c.wavelength = np.linspace(100, 200, 5)
    return c


# ----------------------------------------------------------------------
# TESTS FOR spread_1d_spectrum_to_2d (mode dispatch, counts mismatch)
# ----------------------------------------------------------------------

def test_spread_1d_spectrum_to_2d_dispatches_gaussian():
    """spread_1d_spectrum_to_2d uses Gaussian when spread_y_* are None."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    image, header = spread_1d_spectrum_to_2d(counts, channel, header=[])

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)
    assert header == []


def test_spread_1d_spectrum_to_2d_dispatches_profile():
    """spread_1d_spectrum_to_2d uses profile when spread_y_* are set."""
    channel = _channel_profile()
    counts = np.array([10, 20, 30, 40, 50], dtype=float)

    image, _ = spread_1d_spectrum_to_2d(counts, channel, header=[])

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_spread_1d_spectrum_to_2d_counts_length_mismatch():
    """spread_1d_spectrum_to_2d raises ValueError when len(counts) != nx."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3], dtype=float)  # len=3, nx=5

    with pytest.raises(ValueError, match="Counts length 3 does not match nx 5"):
        spread_1d_spectrum_to_2d(counts, channel, header=[])


def test_spread_1d_spectrum_to_2d_mode_not_implemented():
    """spread_1d_spectrum_to_2d raises NotImplementedError when mode != 1."""
    channel = _channel_gaussian()
    channel.mode = 2
    counts = np.ones(channel.x_pixels)

    with pytest.raises(NotImplementedError, match="mode=2 not implemented"):
        spread_1d_spectrum_to_2d(counts, channel, header=[])


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_gaussian
# ----------------------------------------------------------------------

def test_gaussian_spread_basic():
    """Gaussian spread produces correct shape and column sums match input counts."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    image, _ = _spread_1d_to_2d_gaussian(counts, channel, header=[])

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_gaussian_spread_no_spread_height():
    """spread_half_height_pix <= 0 raises ValueError."""
    channel = _channel_gaussian()
    channel.spread_half_height_pix = 0
    counts = np.ones(channel.x_pixels)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_gaussian(counts, channel, header=[])


def test_gaussian_spread_column_sum_mismatch_raises():
    """When column sums fail validation, _spread_1d_to_2d_gaussian raises ValueError."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    with patch("frame.science.np.allclose", return_value=False):
        with pytest.raises(ValueError, match="Gaussian spread column sum mismatch"):
            _spread_1d_to_2d_gaussian(counts, channel, header=[])


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_profile
# ----------------------------------------------------------------------

def test_profile_spread_basic():
    """Profile spread produces correct shape and column sums match input counts."""
    channel = _channel_profile()
    counts = np.array([10, 20, 30, 40, 50], dtype=float)

    image, _ = _spread_1d_to_2d_profile(counts, channel, header=[])

    assert image.shape == (channel.y_pixels, channel.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_profile_spread_detector_wavelength_mismatch():
    """Detector wavelength length mismatch raises ValueError."""
    channel = _channel_profile()
    channel.wavelength = np.linspace(100, 200, 6)  # wrong length
    counts = np.ones(channel.x_pixels)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_profile(counts, channel, header=[])


def test_profile_spread_out_of_bounds_y_positions():
    """Out-of-bounds y positions are skipped; no crash."""
    channel = _channel_profile()
    channel.spread_y_positions = np.array([100, 200, 300])
    counts = np.ones(channel.x_pixels)

    image, _ = _spread_1d_to_2d_profile(counts, channel, header=[])

    assert image.shape == (channel.y_pixels, channel.x_pixels)


def test_profile_spread_weight_shape_mismatch():
    """Weight shape mismatch (fewer rows than spread positions) raises IndexError."""
    channel = _channel_profile()
    channel.spread_y_weights = np.ones((2, 5))  # 2 rows, 3 spread positions
    counts = np.ones(channel.x_pixels)

    with pytest.raises(IndexError):
        _spread_1d_to_2d_profile(counts, channel, header=[])


# ----------------------------------------------------------------------
# TEST FOR generate_science_frames
# ----------------------------------------------------------------------

def test_generate_science_frames_basic():
    """generate_science_frames produces n_frames; science = dark + detector_image * exptime * gain (dark includes bias)."""
    channel = _channel_gaussian()
    channel.exposure_s = 3.0

    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    base_header = []
    fake_image = np.ones((channel.y_pixels, channel.x_pixels))

    with patch("frame.science.spread_1d_spectrum_to_2d") as mock_spread, \
         patch("frame.science.generate_dark_frame") as mock_dark:

        def fake_spread(counts_in, ch, header):
            return fake_image, header

        mock_spread.side_effect = fake_spread
        mock_dark.return_value = (np.full((channel.y_pixels, channel.x_pixels), 5.0), None)

        frames, headers = generate_science_frames(counts, channel, 2, base_header)

    assert len(frames) == 2
    assert len(headers) == 2
    assert frames[0].shape == (channel.y_pixels, channel.x_pixels)

    hdr_keys = [h[0] for h in headers[0]]
    assert "FILETYPE" in hdr_keys
    assert "CHANNEL" in hdr_keys
    assert "EXP_ID" in hdr_keys
    assert "OBS_ID" in hdr_keys
    assert "EXPTIME" in hdr_keys

    expected = 5.0 + fake_image * 3.0 * channel.ccd_gain
    assert np.allclose(frames[0], expected)


def test_generate_science_frames_n_frames_zero():
    """generate_science_frames with n_frames=0 returns empty frames and headers."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    base_header = []

    frames, headers = generate_science_frames(counts, channel, 0, base_header)

    assert frames == []
    assert headers == []
