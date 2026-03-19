"""Single source of truth for test Channel-like objects. Add new params here so all tests stay green."""

from types import SimpleNamespace

import numpy as np
import pytest

from configs.channel_config import PhotometryChannel


# Defaults for a real PhotometryChannel when tests need isinstance(ch, PhotometryChannel).
# Override in tests (e.g. x_pixels=32, psf_shape=(4, 4)). psf_shape builds psf_image and psf_center_* if not given.
def photometry_channel(**overrides):
    """Build a PhotometryChannel for tests. Override any field. psf_shape=(h,w) sets psf_image and psf_center_*."""
    opts = dict(
        channel_name="NIR",
        x_pixels=20,
        y_pixels=20,
        resolution_factor=1.0,
        dark_noise=0.0,
        dark_current_sigma=0.0,
        read_noise=1.0,
        bias_offset=0.0,
        ccd_gain=1.0,
        exposure_s=1.0,
        n_science_frames=1,
        source_file="test",
        effective_area_file="ea.txt",
        effective_area_wavelength=np.array([1000.0]),
        effective_area=np.array([1.0]),
        pixel_scale=1.0,
        psf_file="psf.txt",
        source_position_x_arcsec=0.0,
        source_position_y_arcsec=0.0,
    )
    psf_shape = overrides.pop("psf_shape", (5, 5))
    opts["psf_image"] = np.ones(psf_shape, dtype=np.float32)
    opts["psf_center_x"] = psf_shape[1] // 2
    opts["psf_center_y"] = psf_shape[0] // 2
    opts.update(overrides)
    return PhotometryChannel(**opts)


@pytest.fixture
def make_channel():
    """Factory fixture for channel-like objects with optional overrides."""
    def _make_channel(**overrides):
        opts = dict(
            channel_name="NUV",
            x_pixels=10,
            y_pixels=8,
            read_noise=5.0,
            bias_offset=100.0,
            dark_noise=0.5,
            dark_current_sigma=2.0,
            ccd_gain=2.0,
            exposure_s=10.0,
            spread_half_height_pix=1,
            mode=1,
            spread_profile_file="dummy.fits",
        )
        opts.update(overrides)
        return SimpleNamespace(**opts)

    return _make_channel
