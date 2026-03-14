import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from tests.loaders._load_channel_test_helpers import _SHARED_CFG


def test_spectroscopy_channel_init():
    slit_w = _SHARED_CFG["slit_width_arcsec"]
    slit_l = _SHARED_CFG["slit_length_arcsec"]
    n_science_frames = _SHARED_CFG["n_science_frames"]
    ch = SpectroscopyChannel(
        channel_name="NUV",
        x_pixels=2048,
        y_pixels=512,
        resolution_factor=1.0,
        dark_noise=0.01,
        dark_current_sigma=0.001,
        read_noise=3.0,
        bias_offset=0.0,
        effective_area_file="ea.txt",
        ccd_gain=1.0,
        exposure_s=10.0,
        mode=1,
        spread_profile_file="",
        spread_half_height_pix=3,
        effective_area_wavelength=np.array([]),
        effective_area=np.array([]),
        pixel_scale=1.0,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        source_file="cfg",
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
        slit_width_arcsec=slit_w,
        slit_length_arcsec=slit_l,
        slit_half_width_arcsec=0.5 * slit_w,
        slit_half_length_arcsec=0.5 * slit_l,
        smear_shift_pixels=10.0,
        slope=0.0,
        intercept_pixels=0.0,
        n_science_frames=n_science_frames,
    )

    assert ch.channel_name == "NUV"
    assert ch.exposure_s == 10.0
    assert ch.x_pixels == 2048


def test_photometry_channel_init():
    n_science_frames = _SHARED_CFG["n_science_frames"]
    ch = PhotometryChannel(
        channel_name="NIR",
        x_pixels=100,
        y_pixels=100,
        resolution_factor=1.0,
        dark_noise=0.0,
        dark_current_sigma=0.0,
        read_noise=1.0,
        bias_offset=0.0,
        ccd_gain=1.0,
        exposure_s=5.0,
        source_file="cfg",
        effective_area_file="ea_ir.txt",
        effective_area_wavelength=np.array([1000.0, 1001.0]),
        effective_area=np.array([0.1, 0.2]),
        pixel_scale=1.0,
        psf_file="nir_psf.txt",
        n_science_frames=n_science_frames,
    )

    assert ch.channel_name == "NIR"
    assert ch.exposure_s == 5.0
    assert ch.psf_file == "nir_psf.txt"
    assert ch.psf_image is None
    assert ch.psf_center_x is None
    assert ch.psf_center_y is None
    assert ch.source_position_x_arcsec is None
    assert ch.source_position_y_arcsec is None


def test_photometry_channel_is_frozen():
    n_science_frames = _SHARED_CFG["n_science_frames"]
    ch = PhotometryChannel(
        channel_name="NIR",
        x_pixels=10,
        y_pixels=10,
        resolution_factor=1.0,
        dark_noise=0.0,
        dark_current_sigma=0.0,
        read_noise=1.0,
        bias_offset=0.0,
        ccd_gain=1.0,
        exposure_s=5.0,
        source_file="cfg",
        effective_area_file="ea_ir.txt",
        effective_area_wavelength=np.array([1000.0]),
        effective_area=np.array([0.1]),
        pixel_scale=1.0,
        psf_file="nir_psf.txt",
        n_science_frames=n_science_frames,
    )

    with pytest.raises(FrozenInstanceError):
        ch.psf_file = "other.txt"