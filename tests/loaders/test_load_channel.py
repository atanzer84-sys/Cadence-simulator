"""
Common parsing/orchestration tests for channel loading.
"""

import numpy as np
import pytest

from configs.channel_config import PhotometryChannel, SpectroscopyChannel
from tests.loaders._load_channel_test_helpers import (
    _BG_LOADER,
    _EA_LOADER,
    _PSF_LOADER,
    _REPO_ROOT,
    _SPREAD_LOADER,
    _UserCfgChannels,
    _fake_psf_image_loader,
    _no_background,
    _no_spread,
    _write_cfg,
    _write_cfg_with_comments,
    load_channel_config,
    load_channels_config,
)


def _assert_error_contains(exc: BaseException, *needles: str) -> None:
    """Assert a raised error message contains stable keyword fragments."""
    msg = str(exc).lower()
    for needle in needles:
        assert needle.lower() in msg


def test_load_channel_config_missing_file_raises(tmp_path):
    """Missing config file raises FileNotFoundError."""
    missing = tmp_path / "missing.cfg"
    with pytest.raises(FileNotFoundError) as exc:
        load_channel_config(missing, exposure_s=10.0)
    _assert_error_contains(exc.value, "config", "not found")


def test_load_channel_config_invalid_int_raises_valueerror(tmp_path):
    """Non-integer x_pixels raises ValueError."""
    cfg_path = tmp_path / "bad_int.cfg"
    _write_cfg(cfg_path, channel_name="NIR", x_pixels="not_an_int")

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)
    _assert_error_contains(exc.value, "x_pixels")


def test_load_channel_config_invalid_float_raises_valueerror(tmp_path):
    """Non-float resolution_factor raises ValueError."""
    cfg_path = tmp_path / "bad_float.cfg"
    _write_cfg(cfg_path, channel_name="NIR", resolution_factor="not_a_float")

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)
    _assert_error_contains(exc.value, "resolution_factor")


def test_load_channel_config_missing_required_key_raises_keyerror(tmp_path):
    """Missing required key (e.g. read_noise) raises KeyError."""
    cfg_path = tmp_path / "missing_key.cfg"
    _write_cfg(
        cfg_path,
        remove_keys={"read_noise"},
        channel_name="NUV",
        x_pixels=2048,
        y_pixels=1024,
        dark_noise=0.01,
        dark_current_sigma=0.001,
    )

    with pytest.raises(KeyError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)
    _assert_error_contains(exc.value, "read_noise")


def test_load_channel_config_ignores_comments(monkeypatch, tmp_path):
    """Full-line and inline comments are ignored during parsing."""
    wl = np.arange(2048, dtype=float)
    ea = np.ones(2048, dtype=float)
    monkeypatch.setattr(_EA_LOADER, lambda _f: (wl, ea, 0.01))
    monkeypatch.setattr(_PSF_LOADER, _fake_psf_image_loader)

    cfg_path = tmp_path / "comments.cfg"
    _write_cfg_with_comments(
        cfg_path,
        channel_name="NIR",
        x_pixels=2048,
        y_pixels=1024,
        read_noise=3.2,
        effective_area_file="ir.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert ch.x_pixels == 2048
    assert ch.read_noise == pytest.approx(3.2)
    assert ch.channel_name == "NIR"


def test_load_channels_config_calls_load_channel_config_three_times(monkeypatch, tmp_path):
    """load_channels_config calls load_channel_config three times for NUV, VIS, IR."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)

    nuv_cfg = tmp_path / "configs" / "waltzer_nuv.cfg"
    vis_cfg = tmp_path / "configs" / "waltzer_vis.cfg"
    ir_cfg = tmp_path / "configs" / "waltzer_nir.cfg"
    nuv_cfg.parent.mkdir(parents=True, exist_ok=True)

    _write_cfg(nuv_cfg, channel_name="NUV", effective_area_file="nuv.txt", x_pixels=2)
    _write_cfg(vis_cfg, channel_name="VIS", effective_area_file="vis.txt", x_pixels=2)
    _write_cfg(
        ir_cfg,
        channel_name="NIR",
        effective_area_file="ir.txt",
        x_pixels=2,
        aperture_pix=4.0,
        psf_file="nir_psf.txt",
    )

    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)
    monkeypatch.setattr(_PSF_LOADER, _fake_psf_image_loader)

    user_cfg = _UserCfgChannels(exposure_NUV_s=3.0, exposure_VIS_s=4.0, exposure_IR_s=5.0)

    nuv_ch, vis_ch, ir_ch = load_channels_config(user_cfg)

    assert isinstance(nuv_ch, SpectroscopyChannel)
    assert isinstance(vis_ch, SpectroscopyChannel)
    assert isinstance(ir_ch, PhotometryChannel)
    assert nuv_ch.channel_name == "NUV"
    assert vis_ch.channel_name == "VIS"
    assert ir_ch.channel_name == "NIR"
