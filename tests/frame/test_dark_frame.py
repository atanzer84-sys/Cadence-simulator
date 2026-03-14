import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame
from frame.dark_frame import generate_dark_frame, generate_dark_frame_with_index
from frame.frame_class import Frame


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
    """generate_dark_frame_with_index creates the correct number of frames; each header has FILETYPE, CHANNEL, EXP_ID, OBS_ID."""
    np.random.seed(0)

    frames = [generate_dark_frame_with_index(channel_cfg, i, base_header) for i in range(3)]

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

    frames = [generate_dark_frame_with_index(channel_cfg, i, base_header) for i in range(2)]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()
    assert mean1 != mean2
    assert std1 != std2
