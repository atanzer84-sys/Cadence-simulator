import numpy as np
import pytest
from astropy.io import fits

from frame.science_frame import generate_science_frame
from frame.frame_class import Frame


@pytest.fixture
def base_header():
    return fits.Header()


def test_generate_science_frame_returns_list_of_one(channel_cfg, base_header):
    """generate_science_frame returns list of 1 frame for pipeline consistency."""
    image = np.ones((channel_cfg.y_pixels, channel_cfg.x_pixels)) * 42.0

    frames = generate_science_frame(image, channel_cfg, base_header)

    assert len(frames) == 1
    assert isinstance(frames[0], Frame)
    assert frames[0].data.shape == (channel_cfg.y_pixels, channel_cfg.x_pixels)


def test_generate_science_frame_header_fields(channel_cfg, base_header):
    """generate_science_frame adds expected FITS header keys."""
    image = np.ones((channel_cfg.y_pixels, channel_cfg.x_pixels))

    frames = generate_science_frame(image, channel_cfg, base_header)

    hdr_keys = list(frames[0].header.keys())
    assert "FILETYPE" in hdr_keys
    assert "CHANNEL" in hdr_keys
    assert "EXP_ID" in hdr_keys
    assert "OBS_ID" in hdr_keys
    assert "EXPTIME" in hdr_keys


def test_generate_science_frame_does_not_modify_image(channel_cfg, base_header):
    """generate_science_frame passes through image data unchanged."""
    image = np.ones((channel_cfg.y_pixels, channel_cfg.x_pixels)) * 42.0

    frames = generate_science_frame(image, channel_cfg, base_header)

    assert np.allclose(frames[0].data, image)
