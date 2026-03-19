import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame, generate_bias_frame_with_index
from frame.frame_class import Frame

import pytest
from dataclasses import replace


@pytest.fixture
def realistic_photometry_channel(make_photometry_channel):
    base = make_photometry_channel()
    return replace(
        base,
        bias_offset=200.0,
        read_noise=3.0,
        ccd_gain=1.0,
    )


# Tests: generate_bias_frame
# Behavior: returns a Frame with correct shape and reasonable mean value
def test_generate_bias_frame_basic_properties(realistic_photometry_channel):
    np.random.seed(0)
    frame = generate_bias_frame(realistic_photometry_channel, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (realistic_photometry_channel.y_pixels, realistic_photometry_channel.x_pixels)

    mean_val = frame.data.mean()
    assert 150 < mean_val < 250


# Tests: generate_bias_frame
# Behavior: populates header with expected bias-related metadata fields
def test_generate_bias_frame_header_fields(realistic_photometry_channel):
    np.random.seed(0)
    header = fits.Header()

    frame = generate_bias_frame(realistic_photometry_channel, header)

    assert isinstance(frame, Frame)
    keys = list(frame.header.keys())

    expected = [
        "MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
        "B_OFFSET", "RNOISE", "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN"
    ]

    for key in expected:
        assert key in keys


# Tests: generate_bias_frame_with_index
# Behavior: produces multiple frames with unique headers containing FILETYPE and CHANNEL
def test_generate_bias_frames_multiple(realistic_photometry_channel, make_header):
    np.random.seed(0)
    frames = [generate_bias_frame_with_index(realistic_photometry_channel, i, make_header()) for i in range(3)]

    assert len(frames) == 3

    for frame in frames:
        assert isinstance(frame, Frame)
        hdr = frame.header
        keys = list(hdr.keys())
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys


# Tests: generate_bias_frame_with_index
# Behavior: independently generated frames differ statistically due to random noise
def test_two_bias_frames_have_different_statistics(realistic_photometry_channel, make_header):
    np.random.seed(42)
    frames = [generate_bias_frame_with_index(realistic_photometry_channel, i, make_header()) for i in range(2)]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()

    assert mean1 != mean2
    assert std1 != std2


# Tests: generate_bias_frame
# Behavior: when header=None, returned Frame.header remains None
def test_generate_bias_frame_header_none(realistic_photometry_channel):
    np.random.seed(0)
    frame = generate_bias_frame(realistic_photometry_channel, header=None)

    assert isinstance(frame, Frame)
    assert frame.header is None


# Tests: generate_bias_frame
# Behavior: returned Frame has correct frame_type="bias" and channel_tag=channel.channel_name
def test_generate_bias_frame_metadata_contract(realistic_photometry_channel):
    np.random.seed(0)
    frame = generate_bias_frame(realistic_photometry_channel, header=None)

    assert frame.frame_type == "bias"
    assert frame.channel_tag == realistic_photometry_channel.channel_name


# Tests: generate_bias_frame
# Behavior: output = generate_bias_image * ccd_gain
def test_generate_bias_frame_gain_scaling(make_photometry_channel, monkeypatch):
    np.random.seed(0)
    channel_cfg = replace(
        make_photometry_channel(),
        bias_offset=0.0,
        read_noise=0.0,
        ccd_gain=2.0,
    )

    fake_bias = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 10.0)
    monkeypatch.setattr("frame.bias_frame.generate_bias_image", lambda ch: fake_bias)

    frame = generate_bias_frame(channel_cfg, header=None)

    expected = fake_bias * channel_cfg.ccd_gain
    np.testing.assert_allclose(frame.data, expected, rtol=0.0, atol=0.0)