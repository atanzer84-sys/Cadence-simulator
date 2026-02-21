import pytest
import numpy as np
from frame.dark import generate_dark_frame, generate_dark_frames


@pytest.fixture
def channel_cfg():
    """
    Minimal fake channel config object containing only the attributes
    used by dark.py. This keeps the test stable and maintainable.
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

    return Cfg()


@pytest.fixture
def base_header():
    """Simple header list used as input to generate_dark_frames."""
    return []


def test_generate_dark_frame_properties(channel_cfg):
    # This test verifies that generate_dark_frame returns a valid 2D array
    # with the expected shape and positive mean values (dark current > 0).
    np.random.seed(0)

    frame, _ = generate_dark_frame(channel_cfg, 10.0, header=[])

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)

    # Dark current should scale with exposure time → mean must be > 0
    assert frame.mean() > 0


def test_generate_dark_frame_header_content(channel_cfg):
    # This test ensures that generate_dark_frame populates the header
    # with all expected metadata keys.
    np.random.seed(0)

    header = []
    _, header = generate_dark_frame(channel_cfg, 10.0, header)

    keys = [k for (k, _, _) in header]

    expected_keys = [
        "MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
        "B_OFFSET", "RNOISE", "DARKSIG", "DARKVAL",
        "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN"
    ]

    for key in expected_keys:
        assert key in keys

def test_generate_dark_frames_multiple(channel_cfg, base_header):
    # This test checks that generate_dark_frames creates the correct number
    # of frames and that each header contains the expected structural keys.
    np.random.seed(0)

    frames, headers = generate_dark_frames(
        channel_cfg,
        n_frames=3,
        exptime_s=10.0,
        base_header=base_header
    )

    assert len(frames) == 3
    assert len(headers) == 3

    for hdr in headers:
        keys = [k for (k, v, c) in hdr]   # ← THE ONLY FIX
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys



def test_two_dark_frames_have_different_statistics(channel_cfg, base_header):
    # This test verifies that two independently generated dark frames
    # have different mean and std values, confirming that noise is applied.
    np.random.seed(42)

    frames, _ = generate_dark_frames(channel_cfg, 2, 10.0, base_header)

    mean1, mean2 = frames[0].mean(), frames[1].mean()
    std1, std2 = frames[0].std(), frames[1].std()

    assert mean1 != mean2
    assert std1 != std2
