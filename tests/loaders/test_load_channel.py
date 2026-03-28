"""
Tests: load_channels_config
Behavior: selects channels from global config flags
"""

from pathlib import Path
import numpy as np
import pytest
from configs.channel_config import PhotometryChannel, SpectroscopyChannel
from loaders.load_channel import (
    _compute_n_science_frames,
    _ensure_effective_area_matches_x_pixels,
    load_channel_config,
    load_channels_config,
)


# Tests: load_channels_config
# Behavior: all channels disabled return no channel objects
def test_load_channels_config_all_channels_disabled_returns_none(
    make_user_config,
    make_global_config,
    monkeypatch,
):
    user_cfg = make_user_config()
    cfg = make_global_config(run_nuv=False, run_vis=False, run_nir=False)
    calls = []

    monkeypatch.setattr("loaders.load_channel.get_global_config", lambda: cfg)
    monkeypatch.setattr("loaders.load_channel.get_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr("loaders.load_channel.load_background_from_global_cfg", lambda: {"bg": "payload"})
    monkeypatch.setattr("loaders.load_channel.load_channel_config", lambda *args: calls.append(args))

    nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg)

    assert nuv_channel is None
    assert vis_channel is None
    assert nir_channel is None
    assert calls == []


# Tests: load_channels_config
# Behavior: only NUV enabled returns only NUV channel
def test_load_channels_config_only_nuv_enabled(
    make_user_config,
    make_global_config,
    monkeypatch,
):
    user_cfg = make_user_config()
    cfg = make_global_config(run_nuv=True, run_vis=False, run_nir=False)
    background = {"bg": "payload"}
    calls = []

    monkeypatch.setattr("loaders.load_channel.get_global_config", lambda: cfg)
    monkeypatch.setattr("loaders.load_channel.get_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr("loaders.load_channel.load_background_from_global_cfg", lambda: background)

    def _load_channel_config(path, exposure_s, bg):
        calls.append((path, exposure_s, bg))
        return {"channel": "NUV"}

    monkeypatch.setattr("loaders.load_channel.load_channel_config", _load_channel_config)

    nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg)

    assert nuv_channel is not None
    assert vis_channel is None
    assert nir_channel is None
    assert len(calls) == 1
    assert calls[0][0] == Path("/repo/configs/waltzer_nuv.cfg")
    assert calls[0][1] == user_cfg.exposure_NUV_s
    assert calls[0][2] is background


# Tests: load_channels_config
# Behavior: only VIS enabled returns only VIS channel
def test_load_channels_config_only_vis_enabled(
    make_user_config,
    make_global_config,
    monkeypatch,
):
    user_cfg = make_user_config()
    cfg = make_global_config(run_nuv=False, run_vis=True, run_nir=False)
    background = {"bg": "payload"}
    calls = []

    monkeypatch.setattr("loaders.load_channel.get_global_config", lambda: cfg)
    monkeypatch.setattr("loaders.load_channel.get_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr("loaders.load_channel.load_background_from_global_cfg", lambda: background)

    def _load_channel_config(path, exposure_s, bg):
        calls.append((path, exposure_s, bg))
        return {"channel": "VIS"}

    monkeypatch.setattr("loaders.load_channel.load_channel_config", _load_channel_config)

    nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg)

    assert nuv_channel is None
    assert vis_channel is not None
    assert nir_channel is None
    assert len(calls) == 1
    assert calls[0][0] == Path("/repo/configs/waltzer_vis.cfg")
    assert calls[0][1] == user_cfg.exposure_VIS_s
    assert calls[0][2] is background


# Tests: load_channels_config
# Behavior: only NIR enabled returns only NIR channel
def test_load_channels_config_only_nir_enabled(
    make_user_config,
    make_global_config,
    monkeypatch,
):
    user_cfg = make_user_config()
    cfg = make_global_config(run_nuv=False, run_vis=False, run_nir=True)
    background = {"bg": "payload"}
    calls = []

    monkeypatch.setattr("loaders.load_channel.get_global_config", lambda: cfg)
    monkeypatch.setattr("loaders.load_channel.get_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr("loaders.load_channel.load_background_from_global_cfg", lambda: background)

    def _load_channel_config(path, exposure_s, bg):
        calls.append((path, exposure_s, bg))
        return {"channel": "NIR"}

    monkeypatch.setattr("loaders.load_channel.load_channel_config", _load_channel_config)

    nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg)

    assert nuv_channel is None
    assert vis_channel is None
    assert nir_channel is not None
    assert len(calls) == 1
    assert calls[0][0] == Path("/repo/configs/waltzer_nir.cfg")
    assert calls[0][1] == user_cfg.exposure_IR_s
    assert calls[0][2] is background



# Tests: _ensure_effective_area_matches_x_pixels
# Behavior: matching effective area length does not raise
def test_ensure_effective_area_matches_x_pixels_matching_length_ok():
    _ensure_effective_area_matches_x_pixels(
        "VIS",
        "ea.txt",
        np.array([1000.0, 1100.0, 1200.0]),
        3,
        "vis.cfg",
    )


# Tests: _ensure_effective_area_matches_x_pixels
# Behavior: mismatched effective area length raises ValueError
def test_ensure_effective_area_matches_x_pixels_mismatch_raises():
    with pytest.raises(ValueError):
        _ensure_effective_area_matches_x_pixels(
            "VIS",
            "ea.txt",
            np.array([1000.0, 1100.0]),
            3,
            "vis.cfg",
        )


# Tests: _compute_n_science_frames
# Behavior: computes frame count from orbit duration and frame time
def test_compute_n_science_frames_returns_integer_frame_count(
    make_global_config,
    monkeypatch,
):
    cfg = make_global_config(orbit_total_duration_s=100.0, readout_gap_s=5.0)

    monkeypatch.setattr("loaders.load_channel.get_global_config", lambda: cfg)

    n_science_frames = _compute_n_science_frames("VIS", 20.0)

    assert n_science_frames == 4


# Tests: load_channel_config
# Behavior: NIR config creates photometry channel
def test_load_channel_config_nir_returns_photometry_channel(monkeypatch):
    raw = {
        "channel_name": "NIR",
        "x_pixels": "4",
        "y_pixels": "5",
        "resolution_factor": "1.0",
        "dark_noise": "0.1",
        "dark_current_sigma": "0.2",
        "read_noise": "0.3",
        "bias_offset": "1.5",
        "ccd_gain": "2.0",
        "effective_area_file": "ea.txt",
        "psf_file": "psf.fits",
        "source_position_x_arcsec": "0.4",
        "source_position_y_arcsec": "0.5",
        "draw_aperture_photometry_overlay": "1",
    }
    background = {
        "background_type": None,
        "background_wavelength": None,
        "background_flux": None,
        "sky_pixel_area_arcsec2": None,
        "zod_dist": None,
        "zod_spectrum_wavelength": None,
        "zod_spectrum_flux": None,
    }

    monkeypatch.setattr("loaders.load_channel.parse_simple_kv", lambda path: raw)
    monkeypatch.setattr(
        "loaders.load_channel.load_effective_area_file",
        lambda filename: (np.array([1000.0, 1100.0]), np.array([0.1, 0.2]), 0.5),
    )
    monkeypatch.setattr(
        "loaders.load_channel.load_psf_image_file",
        lambda filename, channel_name: (np.ones((5, 4)), 2.0, 1.0),
    )
    monkeypatch.setattr("loaders.load_channel._compute_n_science_frames", lambda channel_name, exposure_s: 7)

    channel = load_channel_config(Path("nir.cfg"), 12.0, background)

    assert isinstance(channel, PhotometryChannel)
    assert channel.channel_name == "NIR"
    assert channel.exposure_s == 12.0
    assert channel.n_science_frames == 7
    assert channel.psf_file == "psf.fits"
    assert channel.psf_image.shape == (5, 4)
    assert channel.psf_center_x == 1.0
    assert channel.psf_center_y == 2.0
    assert channel.source_position_x_arcsec == 0.4
    assert channel.source_position_y_arcsec == 0.5
    assert channel.draw_aperture_photometry_overlay is True


# Tests: load_channel_config
# Behavior: VIS config creates spectroscopy channel
def test_load_channel_config_vis_returns_spectroscopy_channel(monkeypatch):
    raw = {
        "channel_name": "VIS",
        "x_pixels": "3",
        "y_pixels": "6",
        "resolution_factor": "1.0",
        "dark_noise": "0.1",
        "dark_current_sigma": "0.2",
        "read_noise": "0.3",
        "bias_offset": "1.5",
        "ccd_gain": "2.0",
        "effective_area_file": "ea.txt",
        "mode": "2",
        "spread_profile_file": "spread.txt",
        "spread_half_height_pix": "4",
        "slit_position_x_arcsec": "0.7",
        "slit_position_y_arcsec": "0.8",
        "slope": "1.1",
        "intercept_pixels": "2.2",
        "slit_width_arcsec": "1.5",
        "slit_length_arcsec": "5.0",
    }
    background = {
        "background_type": "default",
        "background_wavelength": np.array([1000.0, 1100.0]),
        "background_flux": np.array([0.1, 0.2]),
        "sky_pixel_area_arcsec2": 2.5,
        "zod_dist": None,
        "zod_spectrum_wavelength": None,
        "zod_spectrum_flux": None,
    }

    monkeypatch.setattr("loaders.load_channel.parse_simple_kv", lambda path: raw)
    monkeypatch.setattr(
        "loaders.load_channel.load_effective_area_file",
        lambda filename: (np.array([1000.0, 1100.0, 1200.0]), np.array([0.1, 0.2, 0.3]), 0.5),
    )
    monkeypatch.setattr(
        "loaders.load_channel.load_spread_profile_file_spectroscopy",
        lambda filename, channel_name: (
            np.array([0.0, 1.0]),
            np.array(
                [
                    [0.2, 0.2, 0.2],
                    [0.8, 0.8, 0.8],
                ]
            ),
            np.array([1000.0, 1100.0, 1200.0]),
        ),
    )
    monkeypatch.setattr("loaders.load_channel._compute_n_science_frames", lambda channel_name, exposure_s: 9)

    channel = load_channel_config(Path("vis.cfg"), 15.0, background)

    assert isinstance(channel, SpectroscopyChannel)
    assert channel.channel_name == "VIS"
    assert channel.exposure_s == 15.0
    assert channel.n_science_frames == 9
    assert channel.mode == 2
    assert channel.spread_profile_file == "spread.txt"
    assert channel.spread_half_height_pix == 4
    assert np.allclose(channel.spread_y_positions, [0.0, 1.0])
    assert channel.spread_y_weights.ndim == 2
    assert channel.spread_y_weights.shape == (2, 3)
    assert np.allclose(channel.spread_y_weights[:, 0], [0.2, 0.8])
    assert np.allclose(channel.spread_y_wavelengths, [1000.0, 1100.0, 1200.0])
    assert channel.slit_width_arcsec == 1.5
    assert channel.slit_length_arcsec == 5.0
    assert channel.slit_half_width_arcsec == 0.75
    assert channel.slit_half_length_arcsec == 2.5
    assert channel.smear_shift_pixels == 3.0
    assert channel.slope == 1.1
    assert channel.intercept_pixels == 2.2
    assert channel.background_type == "default"
    assert channel.sky_pixel_area_arcsec2 == 2.5