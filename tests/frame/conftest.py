"""Shared fixtures for frame tests."""

import pytest
from astropy.io import fits


@pytest.fixture
def channel_cfg():
    """Minimal channel config shared by bias, dark, science frame tests."""
    class Cfg:
        channel_name = "NUV"
        x_pixels = 10
        y_pixels = 8
        bias_offset = 100.0
        read_noise = 5.0
        dark_current_sigma = 2.0
        dark_noise = 0.5
        ccd_gain = 2.0
        exposure_s = 10.0
        spread_half_height_pix = 1
        mode = 1
        spread_profile_file = "dummy.fits"

    return Cfg()


@pytest.fixture
def base_header():
    """Base FITS header (matches real pipeline usage)."""
    return fits.Header()
