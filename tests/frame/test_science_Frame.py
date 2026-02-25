import numpy as np
from types import SimpleNamespace
from astropy.io import fits

from frame.science_frame import generate_science_frames
from frame.frame_class import Frame


def _channel_gaussian():
    """Channel for Gaussian spread: spread_y_* are None (used by generate_science_frames mocks)."""
    c = SimpleNamespace()
    c.channel_name = "NUV"
    c.x_pixels = 5
    c.y_pixels = 7
    c.mode = 1
    c.spread_half_height_pix = 2
    c.ccd_gain = 2.0
    c.bias_offset = 10.0
    c.read_noise = 3.0
    c.dark_current_sigma = 0.5
    c.dark_noise = 1.0
    c.exposure_s = 3.0
    c.spread_profile_file = "dummy_profile.fits"
    c.wavelength = np.linspace(100, 200, 5)
    c.spread_y_positions = None
    c.spread_y_weights = None
    c.spread_y_wavelengths = None
    return c


def test_generate_science_frames_basic():
    """generate_science_frames wraps a prepared detector image into Frame objects."""
    channel = _channel_gaussian()
    base_header = fits.Header()

    # Pretend this 2D image already contains all detector effects
    fake_image = np.ones((channel.y_pixels, channel.x_pixels)) * 42.0

    frames = generate_science_frames(fake_image, channel, 2, base_header)

    assert len(frames) == 2
    assert isinstance(frames[0], Frame)
    assert frames[0].data.shape == (channel.y_pixels, channel.x_pixels)

    hdr_keys = list(frames[0].header.keys())
    assert "FILETYPE" in hdr_keys
    assert "CHANNEL" in hdr_keys
    assert "EXP_ID" in hdr_keys
    assert "OBS_ID" in hdr_keys
    assert "EXPTIME" in hdr_keys

    # science_frame should not modify the image data it receives
    assert np.allclose(frames[0].data, fake_image)


def test_generate_science_frames_n_frames_zero():
    """generate_science_frames with n_frames=0 returns empty list."""
    channel = _channel_gaussian()
    base_header = fits.Header()
    fake_image = np.ones((channel.y_pixels, channel.x_pixels))

    frames = generate_science_frames(fake_image, channel, 0, base_header)

    assert frames == []
