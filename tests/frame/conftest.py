"""Shared fixtures for frame tests."""

import pytest
from astropy.io import fits

from tests.helpers.channel_factory import channel


@pytest.fixture
def channel_cfg():
    """Minimal channel config shared by bias, dark, science frame tests."""
    return channel()


@pytest.fixture
def base_header():
    """Base FITS header (matches real pipeline usage)."""
    return fits.Header()
