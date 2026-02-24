import pytest
import numpy as np
from astropy.io import fits

from frame.bias import generate_bias_frame, generate_bias_frames
from frame.frame_class import Frame


@pytest.fixture
def channel_cfg():
    """
    Minimal, stable fake channel config object.
    Only includes attributes used by bias.py.
    """

    class Cfg:
        channel_name = "NUV"
        x_pixels = 10
        y_pixels = 8
        bias_offset = 100.0
        read_noise = 5.0
        ccd_gain = 2.0

    return Cfg()


@pytest.fixture
def base_header():
    """Base FITS header used by generate_bias_frames."""
    return fits.Header()


def test_generate_bias_frame_basic_properties(channel_cfg):
    """generate_bias_frame returns a Frame with 2D data of expected shape."""
    np.random.seed(0)

    header = fits.Header()
    frame = generate_bias_frame(channel_cfg, header=header)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)

    # Values should be roughly bias_offset * gain ± noise
    mean_val = frame.data.mean()
    assert 150 < mean_val < 250  # loose bounds, maintainable


def test_generate_bias_frame_header_fields(channel_cfg):
    """generate_bias_frame populates the header with expected statistics and config keys."""
    np.random.seed(0)

    header = fits.Header()
    frame = generate_bias_frame(channel_cfg, header)

    assert isinstance(frame.header, fits.Header)
    keys = list(frame.header.keys())

    expected = [
        "MEAN",
        "MEDIAN",
        "STDDEV",
        "MAX",
        "MIN",
        "B_OFFSET",
        "RNOISE",
        "EXPTIME",
        "YCUT1",
        "YCUT2",
        "CCDGAIN",
    ]

    for key in expected:
        assert key in keys


def test_generate_bias_frames_multiple(channel_cfg, base_header):
    """generate_bias_frames creates the correct number of Frame objects with decorated headers."""
    np.random.seed(0)

    frames = generate_bias_frames(channel_cfg, n_frames=3, base_header=base_header)

    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, Frame)
        assert isinstance(frame.header, fits.Header)
        keys = list(frame.header.keys())
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys


def test_two_bias_frames_have_different_statistics(channel_cfg, base_header):
    """Two independently generated bias frames have different mean/std, confirming noise is applied."""
    np.random.seed(42)

    frames = generate_bias_frames(channel_cfg, n_frames=2, base_header=base_header)

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()

    assert mean1 != mean2
    assert std1 != std2
