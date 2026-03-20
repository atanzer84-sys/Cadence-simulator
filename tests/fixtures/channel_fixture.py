import pytest
import numpy as np
from dataclasses import replace
from configs.channel_config import SpectroscopyChannel, PhotometryChannel


@pytest.fixture
def make_photometry_channel():
    def _make_photometry_channel(**overrides):
        base = dict(
            channel_name="NIR",
            x_pixels=32,
            y_pixels=32,
            resolution_factor=1.0,
            dark_noise=0.0,
            dark_current_sigma=0.0,
            read_noise=1.0,
            bias_offset=0.0,
            ccd_gain=1.0,
            exposure_s=10.0,
            n_science_frames=1,
            source_file="test_source.dat",
            effective_area_file="effective_area.txt",
            effective_area_wavelength=np.array([200.0, 300.0, 400.0], dtype=float),
            effective_area=np.array([0.1, 0.2, 0.3], dtype=float),
            pixel_scale=1.0,
            background_type=None,
            background_wavelength=None,
            background_flux=None,
            sky_pixel_area_arcsec2=None,
            zod_dist=None,
            zod_spectrum_wavelength=None,
            zod_spectrum_flux=None,
        )

        psf_image = np.ones((7, 7), dtype=float)

        specific = dict(
            psf_file="psf.txt",
            psf_image=psf_image,
            psf_center_x=3,
            psf_center_y=3,
            source_position_x_arcsec=0.0,
            source_position_y_arcsec=0.0,
            draw_aperture_photometry_overlay=False,
        )

        base.update(specific)
        base.update(overrides)
        return PhotometryChannel(**base)

    return _make_photometry_channel


@pytest.fixture
def make_spectroscopy_channel():
    def _make_spectroscopy_channel(**overrides):
        base = dict(
            channel_name="NUV",
            x_pixels=64,
            y_pixels=32,
            resolution_factor=1.0,
            dark_noise=0.0,
            dark_current_sigma=0.0,
            read_noise=1.0,
            bias_offset=0.0,
            ccd_gain=1.0,
            exposure_s=10.0,
            n_science_frames=1,
            source_file="test_source.dat",
            effective_area_file="effective_area.txt",
            effective_area_wavelength=np.array([200.0, 250.0, 300.0, 350.0, 400.0], dtype=float),
            effective_area=np.array([0.1, 0.2, 0.3, 0.25, 0.15], dtype=float),
            pixel_scale=1.0,
            background_type=None,
            background_wavelength=None,
            background_flux=None,
            sky_pixel_area_arcsec2=None,
            zod_dist=None,
            zod_spectrum_wavelength=None,
            zod_spectrum_flux=None,
        )

        specific = dict(
            mode=1,
            spread_profile_file="spread_profile.txt",
            spread_half_height_pix=2,
            slit_position_x_arcsec=0.0,
            slit_position_y_arcsec=0.0,
            slope=0.0,
            intercept_pixels=16.0,
            spread_y_positions=None,
            spread_y_weights=None,
            spread_y_wavelengths=None,
            slit_width_arcsec=10.0,
            slit_length_arcsec=60.0,
            slit_half_width_arcsec=5.0,
            slit_half_length_arcsec=30.0,
            smear_shift_pixels=0.0,
        )

        base.update(specific)
        base.update(overrides)
        return SpectroscopyChannel(**base)

    return _make_spectroscopy_channel

@pytest.fixture
def realistic_photometry_channel(make_photometry_channel):
    base = make_photometry_channel()
    return replace(
        base,
        bias_offset=200.0,
        read_noise=3.0,
        ccd_gain=1.0,
    )

@pytest.fixture
def realistic_spectroscopy_channel(make_spectroscopy_channel):
    base = make_spectroscopy_channel()
    return replace(
        base,
        bias_offset=200.0,
        dark_current_sigma=0.02,
        dark_noise=1.0,
        ccd_gain=1.0,
    )