"""
Spectroscopy-specific tests for channel loading.
"""

import numpy as np
import pytest

from configs.channel_config import SpectroscopyChannel
from tests.loaders._load_channel_test_helpers import (
    _BG_LOADER,
    _EA_LOADER,
    _REPO_ROOT_FILES_COMMON,
    _REPO_ROOT_FILES_SPEC,
    _SPREAD_LOADER,
    _ZOD_DIST_LOADER,
    _ZOD_SPECTRUM_LOADER,
    _no_background,
    _no_spread,
    _write_cfg,
    _write_ea_file,
    _write_spread_file,
    load_channel_config,
)


def test_load_channel_config_calls_ea_loader_with_effective_area_file(monkeypatch, tmp_path):
    calls = []

    def _fake_ea(filename: str):
        calls.append(filename)
        return np.array([1000.0, 1100.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv_ea.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert calls == ["nuv_ea.txt"]
    assert isinstance(ch, SpectroscopyChannel)
    assert ch.channel_name == "NUV"


def test_load_channel_config_returns_spectroscopy_channel_with_correct_values(monkeypatch, tmp_path):
    nuv_wl = np.array([1.0, 2.0, 3.0])
    nuv_ea = np.array([0.1, 0.2, 0.3])

    def _fake_ea(filename: str):
        if "nuv" in filename:
            return nuv_wl, nuv_ea, 0.01
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=3)

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert np.allclose(ch.effective_area_wavelength, nuv_wl)
    assert np.allclose(ch.effective_area, nuv_ea)
    assert ch.pixel_scale == pytest.approx(0.01)
    assert ch.channel_name == "NUV"
    assert ch.exposure_s == pytest.approx(5.0)
    assert len(ch.effective_area_wavelength) == 3


def test_load_channel_config_propagates_ea_loader_error(monkeypatch, tmp_path):
    def _fake_ea(filename: str):
        raise ValueError("boom")

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2)

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)

    assert "boom" in str(exc.value)


def test_load_channel_config_raises_if_wavelength_length_does_not_match_x_pixels(monkeypatch, tmp_path):
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 0.01

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, channel_name="NUV")

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)

    msg = str(exc.value)
    assert "NUV" in msg
    assert "nuv.txt" in msg
    assert "len(wavelength)" in msg and "x_pixels" in msg


def test_load_channel_config_sets_spread_fields_from_loader(monkeypatch, tmp_path):
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    def _fake_spread(filename: str, channel_name: str):
        if channel_name == "NUV":
            return np.array([0.0, 1.0]), np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([10.0, 20.0])
        return None, None, None

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _fake_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, spread_profile_file="nuv_spread.txt")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert np.allclose(ch.spread_y_positions, np.array([0.0, 1.0]))
    assert ch.spread_y_weights.shape == (2, 2)
    assert np.allclose(ch.spread_y_wavelengths, np.array([10.0, 20.0]))


def test_load_channel_config_calls_spread_loader_with_empty_filename_and_sets_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _EA_LOADER,
        lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01),
    )

    calls = []

    def _fake_spread(filename: str, channel_name: str):
        calls.append((filename, channel_name))
        return None, None, None

    monkeypatch.setattr(_SPREAD_LOADER, _fake_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, spread_profile_file="")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert calls == [("", "NUV")]
    assert ch.spread_y_positions is None
    assert ch.spread_y_weights is None
    assert ch.spread_y_wavelengths is None


def test_load_channel_config_spread_half_height_pix_optional_none_uses_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, spread_half_height_pix="none")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.spread_half_height_pix == 0


def test_load_channel_config_integration_ea_and_spread_from_real_files(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_FILES_COMMON, lambda: tmp_path)
    monkeypatch.setattr(_REPO_ROOT_FILES_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ea_file(data_dir / "nuv_ea.txt", pixel_scale=0.02, rows=2)
    _write_spread_file(data_dir / "nuv_spread.txt", wavelengths=[10.0, 20.0], num_rows=2)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv_ea.txt",
        spread_profile_file="nuv_spread.txt",
        x_pixels=2,
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert np.allclose(ch.effective_area_wavelength, [1000.0, 1100.0])
    assert np.allclose(ch.effective_area, [0.1, 0.2])
    assert ch.pixel_scale == pytest.approx(0.02)
    assert np.allclose(ch.spread_y_positions, [0.0, 1.0])
    assert ch.spread_y_weights.shape == (2, 2)
    assert np.allclose(ch.spread_y_wavelengths, [10.0, 20.0])
    assert ch.background_wavelength is None
    assert ch.background_flux is None


def test_load_channel_config_sets_background_wavelength_and_flux_from_loader(monkeypatch, tmp_path):
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    bg_wl = np.array([500.0, 600.0, 700.0])
    bg_flux = np.array([0.01, 0.02, 0.03])

    def _fake_bg(filename: str):
        return bg_wl, bg_flux

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _fake_bg)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="default",
        background_file="bg.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert np.allclose(ch.background_wavelength, bg_wl)
    assert np.allclose(ch.background_flux, bg_flux)


def test_load_channel_config_no_background_values_set(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type is None
    assert ch.background_wavelength is None
    assert ch.background_flux is None
    assert ch.zod_dist is None
    assert ch.zod_spectrum_wavelength is None
    assert ch.zod_spectrum_flux is None
    assert ch.sky_pixel_area_arcsec2 == pytest.approx(25.0)


def test_load_channel_config_background_type_empty_or_whitespace_treated_as_none(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    for value in ("", "   ", "  \t  "):
        cfg_path = tmp_path / "nuv.cfg"
        _write_cfg(
            cfg_path,
            effective_area_file="nuv.txt",
            x_pixels=2,
            background_type=value,
        )
        ch = load_channel_config(cfg_path, exposure_s=10.0)
        assert ch.background_type is None
        assert ch.background_wavelength is None
        assert ch.background_flux is None


def test_load_channel_config_background_type_default_calls_background_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)

    bg_calls = []

    def _fake_bg(filename: str):
        bg_calls.append(filename)
        return None, None

    monkeypatch.setattr(_BG_LOADER, _fake_bg)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="default",
        background_file="",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type == "default"
    assert bg_calls == [""]
    assert ch.background_wavelength is None
    assert ch.background_flux is None


def test_load_channel_config_background_type_calc_calls_zod_loaders(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    zod_dist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    zod_wl = np.array([1000.0, 1100.0], dtype=float)
    zod_flux = np.array([0.1, 0.2], dtype=float)

    def _fake_zod_dist(filename: str):
        return zod_dist if filename else None

    def _fake_zod_spectrum(filename: str):
        if filename:
            return zod_wl, zod_flux
        return None, None

    monkeypatch.setattr(_ZOD_DIST_LOADER, _fake_zod_dist)
    monkeypatch.setattr(_ZOD_SPECTRUM_LOADER, _fake_zod_spectrum)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="calc",
        zod_dist_file="zod_dist.txt",
        zod_spectrum_file="zod_spec.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type == "calc"
    assert np.allclose(ch.zod_dist, zod_dist)
    assert np.allclose(ch.zod_spectrum_wavelength, zod_wl)
    assert np.allclose(ch.zod_spectrum_flux, zod_flux)


def test_load_channel_config_background_type_calc_empty_zod_filenames(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    def _fake_zod_dist(filename: str):
        return None

    def _fake_zod_spectrum(filename: str):
        return None, None

    monkeypatch.setattr(_ZOD_DIST_LOADER, _fake_zod_dist)
    monkeypatch.setattr(_ZOD_SPECTRUM_LOADER, _fake_zod_spectrum)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="calc",
        zod_dist_file="",
        zod_spectrum_file="",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type == "calc"
    assert ch.zod_dist is None
    assert ch.zod_spectrum_wavelength is None
    assert ch.zod_spectrum_flux is None


def test_load_channel_config_invalid_background_type_disabled_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="foo",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type is None
    assert ch.background_wavelength is None
    assert ch.background_flux is None
    assert "invalid background_type" in caplog.text.lower()
    assert "foo" in caplog.text.lower()


def test_load_channel_config_sky_pixel_area_default(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.sky_pixel_area_arcsec2 == pytest.approx(25.0)


def test_load_channel_config_sky_pixel_area_override(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        sky_pixel_area_arcsec2=10.5,
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.sky_pixel_area_arcsec2 == pytest.approx(10.5)


def test_load_channel_config_background_type_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, lambda _: (np.array([1.0]), np.array([0.1])))

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(
        cfg_path,
        effective_area_file="nuv.txt",
        x_pixels=2,
        background_type="DEFAULT",
        background_file="bg.txt",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.background_type == "default"
    assert ch.background_wavelength is not None
