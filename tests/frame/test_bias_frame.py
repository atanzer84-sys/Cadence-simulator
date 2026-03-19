import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame, generate_bias_frame_with_index
from frame.frame_class import Frame


# Tests: generate_bias_frame
# Behavior: returns a Frame with correct shape and reasonable mean value
def test_generate_bias_frame_basic_properties(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_bias_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)

    # Bias = offset * gain + noise → mean should fall in a loose expected range
    mean_val = frame.data.mean()
    assert 150 < mean_val < 250


# Tests: generate_bias_frame
# Behavior: populates header with expected bias-related metadata fields
def test_generate_bias_frame_header_fields(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    header = fits.Header()
    frame = generate_bias_frame(channel_cfg, header)

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
def test_generate_bias_frames_multiple(make_channel, base_header):
    np.random.seed(0)
    channel_cfg = make_channel()

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


# Tests: generate_bias_frame_with_index
# Behavior: independently generated frames differ statistically due to random noise
def test_two_bias_frames_have_different_statistics(make_channel, base_header):
    np.random.seed(42)
    channel_cfg = make_channel()

    frames = [generate_bias_frame_with_index(channel_cfg, i, base_header) for i in range(2)]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()

    assert mean1 != mean2
    assert std1 != std2

# Tests: generate_bias_frame
# Behavior: when header=None, returned Frame.header remains None
def test_generate_bias_frame_header_none(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_bias_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.header is None

# Tests: generate_bias_frame
# Behavior: returned Frame has correct frame_type="bias" and channel_tag=channel.channel_name
def test_generate_bias_frame_metadata_contract(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_bias_frame(channel_cfg, header=None)

    assert frame.frame_type == "bias"
    assert frame.channel_tag == channel_cfg.channel_name

# Tests: generate_bias_frame
# Behavior: output = generate_bias_image * ccd_gain
def test_generate_bias_frame_gain_scaling(make_channel, monkeypatch):
    np.random.seed(0)
    channel_cfg = make_channel()

    # Force generate_bias_image to return a known array
    fake_bias = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 10.0)
    monkeypatch.setattr("frame.bias_frame.generate_bias_image", lambda ch: fake_bias)

    frame = generate_bias_frame(channel_cfg, header=None)

    expected = fake_bias * channel_cfg.ccd_gain
    np.testing.assert_allclose(frame.data, expected, rtol=0.0, atol=0.0)


# Tests: generate_bias_frame_with_index
# Behavior: independently generated frames differ due to noise (non-identical arrays)
def test_bias_frames_not_identical(make_channel, base_header):
    np.random.seed(42)
    channel_cfg = make_channel()

    f1 = generate_bias_frame_with_index(channel_cfg, 0, base_header)
    f2 = generate_bias_frame_with_index(channel_cfg, 1, base_header)

    # They should not be identical element‑wise
    assert not np.allclose(f1.data, f2.data, rtol=0.0, atol=0.0)


