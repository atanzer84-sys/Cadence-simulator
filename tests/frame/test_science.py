import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from astropy.io import fits

from frame.science import generate_science_frames
from frame.frame_class import Frame


def _channel_gaussian():
    """Channel for Gaussian spread: spread_y_* are None (used by generate_science_frames mocks)."""
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
    c.spread_y_positions = None
    c.spread_y_weights = None
    c.spread_y_wavelengths = None
    return c


# ----------------------------------------------------------------------
# TESTS FOR generate_science_frames
# ----------------------------------------------------------------------


def test_generate_science_frames_basic():
    """generate_science_frames produces Frames; data = dark + detector_image * exptime * gain (dark includes bias)."""
    channel = _channel_gaussian()
    channel.exposure_s = 3.0

    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    base_header = fits.Header()
    fake_image = np.ones((channel.y_pixels, channel.x_pixels))

    with patch("frame.science.spread_1d_spectrum_to_2d") as mock_spread, \
         patch("frame.science.generate_dark_frame") as mock_dark:

        def fake_spread(counts_in, ch, header):
            return fake_image, header

        mock_spread.side_effect = fake_spread
        # returning ndarray is fine; generate_science_frame converts Frame to ndarray when needed
        mock_dark.return_value = np.full((channel.y_pixels, channel.x_pixels), 5.0)

        frames = generate_science_frames(counts, channel, 2, base_header)

    assert len(frames) == 2
    first = frames[0]
    assert isinstance(first, Frame)
    assert first.data.shape == (channel.y_pixels, channel.x_pixels)

    hdr_keys = list(first.header.keys())
    assert "FILETYPE" in hdr_keys
    assert "CHANNEL" in hdr_keys
    assert "EXP_ID" in hdr_keys
    assert "OBS_ID" in hdr_keys
    assert "EXPTIME" in hdr_keys

    expected = 5.0 + fake_image * 3.0 * channel.ccd_gain
    assert np.allclose(first.data, expected)


def test_generate_science_frames_n_frames_zero():
    """generate_science_frames with n_frames=0 returns empty list."""
    channel = _channel_gaussian()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    base_header = fits.Header()

    frames = generate_science_frames(counts, channel, 0, base_header)

    assert frames == []
