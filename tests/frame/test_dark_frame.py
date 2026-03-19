import numpy as np
from astropy.io import fits
import pytest
from frame.bias_frame import generate_bias_frame
from frame.dark_frame import generate_dark_frame, generate_dark_frame_with_index
from frame.frame_class import Frame


@pytest.fixture
def realistic_spectroscopy_channel(make_spectroscopy_channel):
    return make_spectroscopy_channel(
        bias_offset=200.0,
        dark_current_sigma=0.02,
        dark_noise=1.0,
        read_noise=3.0,
        ccd_gain=1.0,
    )


# Tests: generate_dark_frame
# Behavior: returns a Frame with correct shape and reasonable mean value
def test_generate_dark_frame_properties(realistic_spectroscopy_channel):
    np.random.seed(0)

    frame = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    assert isinstance(frame, Frame)
    assert frame.data.shape == (
        realistic_spectroscopy_channel.y_pixels,
        realistic_spectroscopy_channel.x_pixels,
    )
    assert frame.data.mean() > 0


# Tests: generate_dark_frame
# Behavior: populates header with expected dark-related metadata fields
def test_generate_dark_frame_header_content(realistic_spectroscopy_channel):
    np.random.seed(0)

    header = fits.Header()
    frame = generate_dark_frame(realistic_spectroscopy_channel, header)

    assert isinstance(frame, Frame)
    keys = list(frame.header.keys())

    expected = [
        "MEAN", "MEDIAN", "STDDEV", "MAX", "MIN",
        "B_OFFSET", "RNOISE", "DARKSIG", "DARKVAL",
        "EXPTIME", "YCUT1", "YCUT2", "CCDGAIN",
    ]

    for key in expected:
        assert key in keys


# Tests: generate_dark_frame
# Behavior: when header=None, returned Frame.header remains None
def test_generate_dark_frame_header_none(realistic_spectroscopy_channel):
    np.random.seed(0)

    frame = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    assert isinstance(frame, Frame)
    assert frame.header is None


# Tests: generate_dark_frame
# Behavior: dark frame mean exceeds bias mean (dark = bias + dark current)
def test_dark_frame_mean_exceeds_bias_mean(realistic_spectroscopy_channel):
    np.random.seed(0)
    dark = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    np.random.seed(0)
    bias = generate_bias_frame(realistic_spectroscopy_channel, header=None)

    assert dark.data.mean() > bias.data.mean()


# Tests: generate_dark_frame_with_index
# Behavior: produces multiple frames with unique headers containing FILETYPE and CHANNEL
def test_generate_dark_frames_multiple(realistic_spectroscopy_channel, make_header):
    np.random.seed(0)

    frames = [
        generate_dark_frame_with_index(realistic_spectroscopy_channel, i, make_header())
        for i in range(3)
    ]

    assert len(frames) == 3

    for frame in frames:
        assert isinstance(frame, Frame)
        hdr = frame.header
        keys = list(hdr.keys())
        assert "FILETYPE" in keys
        assert "CHANNEL" in keys
        assert "EXP_ID" in keys
        assert "OBS_ID" in keys


# Tests: generate_dark_frame_with_index
# Behavior: independently generated frames differ statistically due to random noise
def test_two_dark_frames_have_different_statistics(realistic_spectroscopy_channel, make_header):
    np.random.seed(42)

    frames = [
        generate_dark_frame_with_index(realistic_spectroscopy_channel, i, make_header())
        for i in range(2)
    ]

    mean1, mean2 = frames[0].data.mean(), frames[1].data.mean()
    std1, std2 = frames[0].data.std(), frames[1].data.std()

    assert mean1 != mean2
    assert std1 != std2


# Tests: generate_dark_frame
# Behavior: returned Frame has correct frame_type="dark" and channel_tag=channel.channel_name
def test_generate_dark_frame_metadata_contract(realistic_spectroscopy_channel):
    np.random.seed(0)

    frame = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    assert frame.frame_type == "dark"
    assert frame.channel_tag == realistic_spectroscopy_channel.channel_name


# Tests: generate_dark_frame
# Behavior: output = bias + (dark_image * ccd_gain)
def test_generate_dark_frame_gain_scaling(realistic_spectroscopy_channel, monkeypatch):
    np.random.seed(0)

    fake_bias = np.full(
        (
            realistic_spectroscopy_channel.y_pixels,
            realistic_spectroscopy_channel.x_pixels,
        ),
        100.0,
    )
    fake_dark = np.full(
        (
            realistic_spectroscopy_channel.y_pixels,
            realistic_spectroscopy_channel.x_pixels,
        ),
        2.0,
    )

    monkeypatch.setattr(
        "frame.dark_frame.generate_bias_frame",
        lambda ch, header=None: Frame(fake_bias, None, "bias", ch.channel_name),
    )
    monkeypatch.setattr(
        "frame.dark_frame.generate_dark_image",
        lambda ch: fake_dark,
    )

    frame = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    expected = fake_bias + fake_dark * realistic_spectroscopy_channel.ccd_gain
    np.testing.assert_allclose(frame.data, expected, rtol=0.0, atol=0.0)


# Tests: generate_dark_frame
# Behavior: dark frame exceeds bias frame when dark image is positive
def test_dark_frame_exceeds_bias_when_dark_image_positive(realistic_spectroscopy_channel, monkeypatch):
    fake_bias = np.full(
        (
            realistic_spectroscopy_channel.y_pixels,
            realistic_spectroscopy_channel.x_pixels,
        ),
        100.0,
    )
    fake_dark = np.full(
        (
            realistic_spectroscopy_channel.y_pixels,
            realistic_spectroscopy_channel.x_pixels,
        ),
        1.0,
    )

    monkeypatch.setattr(
        "frame.dark_frame.generate_bias_frame",
        lambda ch, header=None: Frame(fake_bias, None, "bias", ch.channel_name),
    )
    monkeypatch.setattr(
        "frame.dark_frame.generate_dark_image",
        lambda ch: fake_dark,
    )

    frame = generate_dark_frame(realistic_spectroscopy_channel, header=None)

    assert np.all(frame.data > fake_bias)


# Tests: generate_dark_frame_with_index
# Behavior: header contains FILETYPE, CHANNEL, EXP_ID, OBS_ID
def test_generate_dark_frame_with_index_header_values(realistic_spectroscopy_channel, make_header):
    np.random.seed(0)

    frame = generate_dark_frame_with_index(realistic_spectroscopy_channel, 0, make_header())
    header = frame.header

    for key in ["FILETYPE", "CHANNEL", "EXP_ID", "OBS_ID"]:
        assert key in header