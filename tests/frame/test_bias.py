import pytest
import numpy as np
from frame.bias import generate_bias_frame, generate_bias_frames


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
    """A simple mutable header list used by generate_bias_frames."""
    return []


def test_generate_bias_frame_basic_properties(channel_cfg):
    # This test checks that generate_bias_frame returns an array of correct shape
    # and that the values are within a reasonable range around the bias offset.
    np.random.seed(0)

    frame, _ = generate_bias_frame(channel_cfg, header=[])

    assert frame.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)
    assert isinstance(frame, np.ndarray)

    # Values should be roughly bias_offset * gain ± noise
    mean_val = frame.mean()
    assert 150 < mean_val < 250  # loose bounds, maintainable


def test_generate_bias_frame_header_fields(channel_cfg):
    # This test ensures that generate_bias_frame adds the expected header keys.
    np.random.seed(0)

    header = []
    _, header = generate_bias_frame(channel_cfg, header)

    keys = [k for (k, _, _) in header]

    expected = ["MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
                "B_OFFSET", "RNOISE", "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN"]

    for key in expected:
        assert key in keys


def test_generate_bias_frames_multiple(channel_cfg, base_header):
    # This test verifies that generate_bias_frames creates the correct number
    # of frames and headers, and that each header contains FILETYPE and CHANNEL.
    np.random.seed(0)

    frames, headers = generate_bias_frames(channel_cfg, n_frames=3, base_header=base_header)

    assert len(frames) == 3
    assert len(headers) == 3

    for hdr in headers:
        keys = [k for (k, _, _) in hdr]
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys

def test_two_bias_frames_have_different_statistics(channel_cfg, base_header):
    # This test verifies that two independently generated bias frames
    # have different mean and std values, confirming that noise is applied.
    np.random.seed(42)

    frames, _ = generate_bias_frames(channel_cfg, n_frames=2, base_header=base_header)

    mean1, mean2 = frames[0].mean(), frames[1].mean()
    std1, std2 = frames[0].std(), frames[1].std()

    assert mean1 != mean2
    assert std1 != std2
