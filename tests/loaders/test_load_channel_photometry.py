"""
Photometry-specific tests for channel loading.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from configs.channel_config import PhotometryChannel
from tests.loaders._load_channel_test_helpers import (
    _EA_LOADER,
    _PSF_LOADER,
    _SPREAD_LOADER,
    _fake_psf_image_loader,
    _write_cfg,
    load_channel_config,
)


def test_channel_is_frozen(monkeypatch, tmp_path):
    """SpectroscopyChannel and PhotometryChannel are immutable (frozen dataclass)."""
    monkeypatch.setattr(_EA_LOADER, lambda _f: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_PSF_LOADER, _fake_psf_image_loader)

    cfg_path = tmp_path / "ir.cfg"
    _write_cfg(
        cfg_path,
        channel_name="NIR",
        x_pixels=2,
        effective_area_file="ir.txt",
        aperture_pix=4.0,
        psf_file="nir_psf.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    with pytest.raises(FrozenInstanceError):
        ch.x_pixels = 200


def test_load_channel_config_duplicate_keys_last_wins(monkeypatch, tmp_path):
    """When a key appears multiple times, the last occurrence overwrites earlier ones."""
    wl = np.arange(2048, dtype=float)
    ea = np.ones(2048, dtype=float)
    monkeypatch.setattr(_EA_LOADER, lambda _f: (wl, ea, 0.01))
    monkeypatch.setattr(_PSF_LOADER, _fake_psf_image_loader)

    cfg_path = tmp_path / "channel.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="ea.txt",
        x_pixels=1024,
        y_pixels=512,
        dark_noise=0.0,
        read_noise=3.5,
        channel_name="NIR",
        dark_current_sigma=0.001,
    )
    with cfg_path.open("a", encoding="utf-8") as f:
        f.write("\nx_pixels = 2048\n")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.x_pixels == 2048


def test_load_channel_config_ir_returns_photometry_channel(monkeypatch, tmp_path):
    """load_channel_config for IR returns PhotometryChannel and loads EA/PSF inputs."""
    ea_calls = []

    def _fake_ea(filename: str):
        ea_calls.append(filename)
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, lambda *_args: (None, None, None))
    monkeypatch.setattr(_PSF_LOADER, _fake_psf_image_loader)

    cfg_path = tmp_path / "ir.cfg"
    _write_cfg(
        cfg_path,
        channel_name="NIR",
        effective_area_file="ir.txt",
        x_pixels=2,
        aperture_pix=4.0,
        psf_file="nir_psf.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert isinstance(ch, PhotometryChannel)
    assert ch.channel_name == "NIR"
    assert ch.exposure_s == pytest.approx(5.0)
    assert ea_calls == ["ir.txt"]
