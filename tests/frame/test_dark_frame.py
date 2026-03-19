import numpy as np
from astropy.io import fits
from frame.bias_frame import generate_bias_frame
from frame.dark_frame import generate_dark_frame, generate_dark_frame_with_index
from frame.frame_class import Frame


def test_generate_dark_frame_properties(make_channel):
    """generate_dark_frame returns a valid Frame with 2D data and positive mean (dark current > 0)."""
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_dark_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)
    assert frame.data.mean() > 0


def test_generate_dark_frame_header_content(make_channel):
    """generate_dark_frame populates the header with all expected metadata keys."""
    np.random.seed(0)
    channel_cfg = make_channel()

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


def test_generate_dark_frame_header_none_returns_none(make_channel):
    """generate_dark_frame(channel, header=None) returns Frame with header=None and does not populate header."""
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_dark_frame(channel_cfg, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)
    assert frame.header is None


def test_dark_frame_mean_exceeds_bias_mean(make_channel):
    """Dark frame mean exceeds bias mean (dark = bias + dark_current contribution)."""
    np.random.seed(0)
    channel_cfg = make_channel()
    dark = generate_dark_frame(channel_cfg, header=None)

    np.random.seed(0)
    bias = generate_bias_frame(channel_cfg, header=None)

    assert dark.data.mean() > bias.data.mean()


def test_generate_dark_frames_multiple(make_channel, base_header):
    """generate_dark_frame_with_index creates the correct number of frames; each header has FILETYPE, CHANNEL, EXP_ID, OBS_ID."""
    np.random.seed(0)
    channel_cfg = make_channel()

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



def test_two_dark_frames_have_different_statistics(make_channel, base_header):
    """Two independently generated dark frames have different mean/std, confirming noise is applied."""
    np.random.seed(42)
    channel_cfg = make_channel()

    frames = [generate_dark_frame_with_index(channel_cfg, i, base_header) for i in range(2)]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()
    assert mean1 != mean2
    assert std1 != std2


# Tests: generate_dark_frame
# Behavior: returned Frame has correct frame_type="dark" and channel_tag=channel.channel_name
def test_generate_dark_frame_metadata_contract(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_dark_frame(channel_cfg, header=None)

    assert frame.frame_type == "dark"
    assert frame.channel_tag == channel_cfg.channel_name


# Tests: generate_dark_frame
# Behavior: dark = bias + (dark_image * ccd_gain) — gain scaling is applied correctly
def test_generate_dark_frame_gain_scaling(make_channel, monkeypatch):
    np.random.seed(0)
    channel_cfg = make_channel()

    fake_bias = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 100.0)
    fake_dark = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 2.0)

    monkeypatch.setattr(
        "frame.dark_frame.generate_bias_frame",
        lambda ch, header=None: Frame(fake_bias, None, "bias", ch.channel_name)
    )
    monkeypatch.setattr(
        "frame.dark_frame.generate_dark_image",
        lambda ch: fake_dark
    )

    frame = generate_dark_frame(channel_cfg, header=None)

    expected = fake_bias + fake_dark * channel_cfg.ccd_gain
    np.testing.assert_allclose(frame.data, expected, rtol=0.0, atol=0.0)

# Tests: generate_dark_frame
# Behavior: header contains all expected metadata fields written by production
def test_generate_dark_frame_header_values(make_channel):
    np.random.seed(0)
    channel_cfg = make_channel()

    header = fits.Header()
    generate_dark_frame(channel_cfg, header)

    expected_keys = [
        "MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
        "B_OFFSET", "RNOISE", "DARKSIG", "DARKVAL",
        "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN",
    ]

    for key in expected_keys:
        assert key in header


# Tests: generate_dark_frame
# Behavior: dark frame exceeds bias frame when dark_image is positive (deterministic test)
def test_dark_frame_exceeds_bias_when_dark_image_positive(make_channel, monkeypatch):
    channel_cfg = make_channel()
    fake_bias = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 100.0)
    fake_dark = np.full((channel_cfg.y_pixels, channel_cfg.x_pixels), 1.0)

    monkeypatch.setattr(
        "frame.dark_frame.generate_bias_frame",
        lambda ch, header=None: Frame(fake_bias, None, "bias", ch.channel_name)
    )
    monkeypatch.setattr(
        "frame.dark_frame.generate_dark_image",
        lambda ch: fake_dark
    )

    frame = generate_dark_frame(channel_cfg, header=None)

    assert np.all(frame.data > fake_bias)

# Tests: generate_dark_frame_with_index
# Behavior: header contains FILETYPE, CHANNEL, EXP_ID, OBS_ID
def test_generate_dark_frame_with_index_header_values(make_channel, base_header):
    np.random.seed(0)
    channel_cfg = make_channel()

    frame = generate_dark_frame_with_index(channel_cfg, 0, base_header)
    header = frame.header

    for key in ["FILETYPE", "CHANNEL", "EXP_ID", "OBS_ID"]:
        assert key in header
