import numpy as np
import pytest
from unittest.mock import patch

from frame.science import (
    generate_science_frames,
    _spread_1d_to_2d_gaussian,
    _spread_1d_to_2d_profile,
)

# ----------------------------------------------------------------------
# SHARED DUMMY CONFIGS (used by ALL tests)
# ----------------------------------------------------------------------

class DummyCfg:
    def __init__(self):
        self.channel_name = "NUV"
        self.x_pixels = 5
        self.y_pixels = 7
        self.mode = 1
        self.spread_half_height_pix = 2
        self.ccd_gain = 2.0
        self.bias_offset = 10.0
        self.read_noise = 3.0
        self.dark_current_sigma = 0.5
        self.dark_noise = 1.0
        self.spread_profile_file = "dummy_profile.fits" 


class DummyCal:
    def __init__(self):
        self.spread_y_positions = None
        self.spread_y_weights = None
        self.spread_y_wavelengths = None
        self.wavelength = np.linspace(100, 200, 5)


class DummyCalProfile:
    def __init__(self):
        self.wavelength = np.linspace(100, 200, 5)
        self.spread_y_positions = np.array([-1, 0, 1])
        self.spread_y_weights = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.6, 0.6, 0.6, 0.6, 0.6],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ])
        self.spread_y_wavelengths = np.linspace(100, 200, 5)


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_gaussian
# ----------------------------------------------------------------------

def test_gaussian_spread_basic():
    # This test checks that the Gaussian spread produces the correct shape
    # and that the column sums match the input counts.
    cfg = DummyCfg()
    cal = DummyCal()

    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    image, header = _spread_1d_to_2d_gaussian(counts, cfg, cal, header=[])

    assert image.shape == (cfg.y_pixels, cfg.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_gaussian_spread_no_spread_height():
    # spread_half_height_pix <= 0 must raise ValueError
    cfg = DummyCfg()
    cfg.spread_half_height_pix = 0
    cal = DummyCal()
    counts = np.ones(cfg.x_pixels)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_gaussian(counts, cfg, cal, header=[])


def test_gaussian_spread_column_mismatch_error():
    # Force mismatch by modifying the output after creation
    cfg = DummyCfg()
    cal = DummyCal()
    counts = np.array([1, 2, 3, 4, 5], dtype=float)

    image, header = _spread_1d_to_2d_gaussian(counts, cfg, cal, header=[])

    # Break column sums artificially
    image[:, 0] *= 2

    col_sums = image.sum(axis=0)
    assert not np.allclose(col_sums, counts)


# ----------------------------------------------------------------------
# TESTS FOR _spread_1d_to_2d_profile
# ----------------------------------------------------------------------

def test_profile_spread_basic():
    # This test checks that the profile spread produces the correct shape
    # and that the column sums match the input counts.
    cfg = DummyCfg()
    cal = DummyCalProfile()

    counts = np.array([10, 20, 30, 40, 50], dtype=float)

    image, header = _spread_1d_to_2d_profile(counts, cfg, cal, header=[])

    assert image.shape == (cfg.y_pixels, cfg.x_pixels)
    assert np.allclose(image.sum(axis=0), counts)


def test_profile_spread_detector_wavelength_mismatch():
    # Detector wavelength length mismatch must raise ValueError
    cfg = DummyCfg()
    cal = DummyCalProfile()
    cal.wavelength = np.linspace(100, 200, 6)  # wrong length
    counts = np.ones(cfg.x_pixels)

    with pytest.raises(ValueError):
        _spread_1d_to_2d_profile(counts, cfg, cal, header=[])


def test_profile_spread_out_of_bounds_y_positions():
    # Out-of-bounds y positions should not crash — they are skipped
    cfg = DummyCfg()
    cal = DummyCalProfile()

    cal.spread_y_positions = np.array([100, 200, 300])
    counts = np.ones(cfg.x_pixels)

    image, header = _spread_1d_to_2d_profile(counts, cfg, cal, header=[])

    assert image.shape == (cfg.y_pixels, cfg.x_pixels)


def test_profile_spread_weight_shape_mismatch():
    # Weight shape mismatch must raise an exception
    cfg = DummyCfg()
    cal = DummyCalProfile()

    cal.spread_y_weights = np.ones((2, 5))  # wrong shape
    counts = np.ones(cfg.x_pixels)

    with pytest.raises(Exception):
        _spread_1d_to_2d_profile(counts, cfg, cal, header=[])


# ----------------------------------------------------------------------
# TEST FOR generate_science_frames
# ----------------------------------------------------------------------

def test_generate_science_frames_basic():
    # This test checks that generate_science_frames produces the correct number
    # of frames and headers, and that the science frame is constructed from
    # bias + dark + detector_image * exposure_time * gain.

    channel_cfg = DummyCfg()
    channel_cal = DummyCal()

    counts = np.array([1, 2, 3, 4, 5], dtype=float)
    base_header = []

    fake_image = np.ones((channel_cfg.y_pixels, channel_cfg.x_pixels))

    # Mock the heavy functions so the test is deterministic
    with patch("frame.science.spread_1d_spectrum_to_2d") as mock_spread, \
         patch("frame.science.generate_bias_frame") as mock_bias, \
         patch("frame.science.generate_dark_frame") as mock_dark:

        # IMPORTANT: return the SAME header, not a new one
        def fake_spread_func(counts, cfg, cal, header):
            return fake_image, header

        mock_spread.side_effect = fake_spread_func

        mock_bias.return_value = (np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 10.0), None)
        mock_dark.return_value = (np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 5.0), None)

        frames, headers = generate_science_frames(
            counts,
            channel_cfg,
            channel_cal,
            n_frames=2,
            exposure_time_s=3.0,
            base_header=base_header
        )

    # Check number of frames
    assert len(frames) == 2
    assert len(headers) == 2

    # Check frame shape
    assert frames[0].shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)

    # Check header contains expected keys
    hdr_keys = [h[0] for h in headers[0]]
    assert "FILETYPE" in hdr_keys
    assert "CHANNEL" in hdr_keys
    assert "EXP_ID" in hdr_keys
    assert "OBS_ID" in hdr_keys
    assert "EXPTIME" in hdr_keys

    # Check science frame math:
    expected = 10.0 + 5.0 + fake_image * 3.0 * channel_cfg.ccd_gain
    assert np.allclose(frames[0], expected)
