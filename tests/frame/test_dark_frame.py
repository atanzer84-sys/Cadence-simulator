import pytest
import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame
from frame.dark_frame import generate_dark_frame, generate_dark_frames
from frame.frame_class import Frame


@pytest.fixture
def channel_cfg():
    """
    Minimal fake channel config with only the attributes used by dark.py.
    Refactor-safe: add new attributes here when dark.py starts using them.
    """
    class Cfg:
        channel_name = "NUV"
        x_pixels = 10
        y_pixels = 8
        bias_offset = 100.0
        read_noise = 5.0
        dark_current_sigma = 2.0
        dark_noise = 0.5
        ccd_gain = 2.0
        exposure_s = 10.0  # used by generate_dark_frame (no longer passed as arg)

    return Cfg()


@pytest.fixture
def base_header():
    """Base FITS header used as input to generate_dark_frames (matches real pipeline usage)."""
    return fits.Header()


def test_generate_dark_frame_properties(channel_cfg):
    """generate_dark_frame returns a valid Frame with 2D data and positive mean (dark current > 0)."""
    np.random.seed(0)

    frame = generate_dark_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)
    assert frame.data.mean() > 0


def test_generate_dark_frame_header_content(channel_cfg):
    """generate_dark_frame populates the header with all expected metadata keys."""
    np.random.seed(0)

    header = fits.Header()
    frame = generate_dark_frame(channel_cfg, header)

    assert isinstance(frame, Frame)
    keys = list(frame.header.keys())
    expected_keys = [
        "MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
        "B_OFFSET", "RNOISE", "DARKSIG", "DARKVAL",
        "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN",
    ]
    for key in expected_keys:
        assert key in keys


def test_generate_dark_frame_header_none_returns_none(channel_cfg):
    """generate_dark_frame(channel, header=None) returns Frame with header=None and does not populate header."""
    np.random.seed(0)

    frame = generate_dark_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)
    assert frame.header is None


def test_dark_frame_mean_exceeds_bias_mean(channel_cfg):
    """Dark frame mean exceeds bias mean (dark = bias + dark_current contribution)."""
    np.random.seed(0)
    dark = generate_dark_frame(channel_cfg, header=None)

    np.random.seed(0)
    bias = generate_bias_frame(channel_cfg, header=None)

    assert dark.data.mean() > bias.data.mean()


def test_generate_dark_frames_multiple(channel_cfg, base_header):
    """generate_dark_frames creates the correct number of frames; each header has FILETYPE, CHANNEL, EXP_ID, OBS_ID."""
    np.random.seed(0)

    frames = generate_dark_frames(channel_cfg, 3, base_header)

    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, Frame)
        hdr = frame.header
        keys = list(hdr.keys())
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys



def test_two_dark_frames_have_different_statistics(channel_cfg, base_header):
    """Two independently generated dark frames have different mean/std, confirming noise is applied."""
    np.random.seed(42)

    frames = generate_dark_frames(channel_cfg, 2, base_header)

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()
    assert mean1 != mean2
    assert std1 != std2
