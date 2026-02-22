import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from frame.science import generate_science_frames


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
