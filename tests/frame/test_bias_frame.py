import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame, generate_bias_frame_with_index
from frame.frame_class import Frame


def test_generate_bias_frame_basic_properties(channel_cfg):
    # This test checks that generate_bias_frame returns a Frame with correct shape
    # and that the values are within a reasonable range around the bias offset.
    np.random.seed(0)

    frame = generate_bias_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)

    # Values should be roughly bias_offset * gain ± noise
    mean_val = frame.data.mean()
    assert 150 < mean_val < 250  # loose bounds, maintainable

def test_generate_bias_frame_header_fields(channel_cfg):
    # This test ensures that generate_bias_frame adds the expected header keys.
    np.random.seed(0)

    header = fits.Header()
    frame = generate_bias_frame(channel_cfg, header)

    assert isinstance(frame, Frame)
    keys = list(frame.header.keys())

    expected = ["MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
                "B_OFFSET", "RNOISE", "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN"]

    for key in expected:
        assert key in keys

def test_generate_bias_frames_multiple(channel_cfg, base_header):
    # This test verifies that generate_bias_frame_with_index creates the correct number
    # of frames and headers, and that each header contains FILETYPE and CHANNEL.
    np.random.seed(0)

    frames = [generate_bias_frame_with_index(channel_cfg, i, base_header) for i in range(3)]

    assert len(frames) == 3

    for frame in frames:
        assert isinstance(frame, Frame)
        hdr = frame.header
        keys = list(hdr.keys())
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys

def test_two_bias_frames_have_different_statistics(channel_cfg, base_header):
    # This test verifies that two independently generated bias frames
    # have different mean and std values, confirming that noise is applied.
    np.random.seed(42)

    frames = [generate_bias_frame_with_index(channel_cfg, i, base_header) for i in range(2)]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()

    assert mean1 != mean2
    assert std1 != std2
