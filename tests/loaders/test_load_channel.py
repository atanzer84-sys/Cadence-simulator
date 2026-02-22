"""
Tests for load_channel_config and load_channels_config.
Migrated from tests/instrument/test_detector.py (calibration/loading tests).
"""
import numpy as np
import pytest
from pathlib import Path

from configs.channel import SpectroscopyChannel, PhotometryChannel
from loaders.load_channel import load_channel_config, load_channels_config


def _write_cfg(path: Path, **kwargs) -> None:
    """Write a minimal NUV/VIS channel config file."""
    defaults = {
        "channel_name": "NUV",
        "x_pixels": 2,
        "y_pixels": 10,
        "resolution_factor": 1.0,
        "dark_noise": 0.01,
        "dark_current_sigma": 0.001,
        "read_noise": 3.0,
        "effective_area_file": "nuv.txt",
        "bias_offset": 0.0,
        "ccd_gain": 1.0,
        "mode": 1,
        "spread_profile_file": "",
        "spread_half_height_pix": 2,
    }
    defaults.update(kwargs)
    lines = [f"{k} = {v}" for k, v in defaults.items()]
    path.write_text("\n".join(lines), encoding="utf-8")


def _no_spread(_filename: str, _channel_name: str):
    """Return None spread data so tests don't call the real loader."""
    return None, None, None


# ----------------------------------------------------------------------
# load_channel_config: loader calls and return values
# ----------------------------------------------------------------------


def test_load_channel_config_calls_ea_loader_with_effective_area_file(monkeypatch, tmp_path):
    """load_channel_config calls load_effective_area_file with effective_area_file from cfg."""
    calls = []

    def _fake_ea(filename: str):
        calls.append(filename)
        return np.array([1000.0, 1100.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv_ea.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert calls == ["nuv_ea.txt"]
    assert isinstance(ch, SpectroscopyChannel)
    assert ch.channel_name == "NUV"


def test_load_channel_config_returns_spectroscopy_channel_with_correct_values(monkeypatch, tmp_path):
    """load_channel_config returns SpectroscopyChannel with wavelength, effective_area, pixel_scale from loader."""
    nuv_wl = np.array([1.0, 2.0, 3.0])
    nuv_ea = np.array([0.1, 0.2, 0.3])

    def _fake_ea(filename: str):
        if "nuv" in filename:
            return nuv_wl, nuv_ea, 0.01
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=3)

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert np.allclose(ch.wavelength, nuv_wl)
    assert np.allclose(ch.effective_area, nuv_ea)
    assert ch.pixel_scale == pytest.approx(0.01)
    assert ch.channel_name == "NUV"
    assert ch.exposure_s == pytest.approx(5.0)


def test_load_channel_config_propagates_ea_loader_error(monkeypatch, tmp_path):
    """load_channel_config propagates ValueError from load_effective_area_file."""
    def _fake_ea(filename: str):
        raise ValueError("boom")

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2)

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)

    assert "boom" in str(exc.value)


def test_load_channel_config_raises_if_wavelength_length_does_not_match_x_pixels(monkeypatch, tmp_path):
    """load_channel_config raises ValueError when len(wavelength) != x_pixels."""
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 0.01

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, channel_name="NUV")

    with pytest.raises(ValueError) as exc:
        load_channel_config(cfg_path, exposure_s=10.0)

    assert "NUV:" in str(exc.value)
    assert "nuv.txt" in str(exc.value)
    assert "len(wavelength)=3 != x_pixels=2" in str(exc.value)


def test_load_channel_config_succeeds_when_lengths_match(monkeypatch, tmp_path):
    """load_channel_config succeeds when wavelength grid length matches x_pixels."""
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.channel_name == "NUV"
    assert len(ch.wavelength) == 2


def test_load_channel_config_sets_spread_fields_from_loader(monkeypatch, tmp_path):
    """load_channel_config stores spread_y_positions, spread_y_weights, spread_y_wavelengths from spread loader."""
    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    def _fake_spread(filename: str, channel_name: str):
        if channel_name == "NUV":
            return np.array([0.0, 1.0]), np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([10.0, 20.0])
        return None, None, None

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _fake_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, spread_profile_file="nuv_spread.txt")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert np.allclose(ch.spread_y_positions, np.array([0.0, 1.0]))
    assert ch.spread_y_weights.shape == (2, 2)
    assert np.allclose(ch.spread_y_wavelengths, np.array([10.0, 20.0]))


def test_load_channel_config_calls_spread_loader_with_empty_filename_and_sets_none(monkeypatch, tmp_path):
    """Empty spread_profile_file results in None spread fields."""
    monkeypatch.setattr(
        "loaders.load_channel.load_effective_area_file",
        lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01),
    )

    calls = []

    def _fake_spread(filename: str, channel_name: str):
        calls.append((filename, channel_name))
        return None, None, None

    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _fake_spread)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, spread_profile_file="")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert calls == [("", "NUV")]
    assert ch.spread_y_positions is None
    assert ch.spread_y_weights is None
    assert ch.spread_y_wavelengths is None


# ----------------------------------------------------------------------
# load_channels_config: three-channel orchestration
# ----------------------------------------------------------------------


def test_load_channels_config_calls_load_channel_config_three_times(monkeypatch, tmp_path):
    """load_channels_config calls load_channel_config three times for NUV, VIS, IR."""
    monkeypatch.setattr("loaders.load_channel.get_repo_root", lambda: tmp_path)

    nuv_cfg = tmp_path / "configs" / "waltzer_nuv.cfg"
    vis_cfg = tmp_path / "configs" / "waltzer_vis.cfg"
    ir_cfg = tmp_path / "configs" / "waltzer_ir.cfg"
    nuv_cfg.parent.mkdir(parents=True, exist_ok=True)

    _write_cfg(nuv_cfg, channel_name="NUV", effective_area_file="nuv.txt", x_pixels=2)
    _write_cfg(vis_cfg, channel_name="VIS", effective_area_file="vis.txt", x_pixels=2)
    _write_cfg(ir_cfg, channel_name="IR", effective_area_file="ir.txt", x_pixels=2)

    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    user_cfg = type("UserCfg", (), {"exposure_NUV_s": 3.0, "exposure_VIS_s": 4.0, "exposure_IR_s": 5.0})()

    nuv_ch, vis_ch, ir_ch = load_channels_config(user_cfg)

    assert isinstance(nuv_ch, SpectroscopyChannel)
    assert isinstance(vis_ch, SpectroscopyChannel)
    assert isinstance(ir_ch, PhotometryChannel)
    assert nuv_ch.channel_name == "NUV"
    assert vis_ch.channel_name == "VIS"
    assert ir_ch.channel_name == "IR"


# ----------------------------------------------------------------------
# IR channel: PhotometryChannel, no effective area loading
# ----------------------------------------------------------------------


def test_load_channel_config_ir_returns_photometry_channel_without_loading_ea(monkeypatch, tmp_path):
    """load_channel_config for IR returns PhotometryChannel without calling load_effective_area_file."""
    ea_calls = []

    def _fake_ea(filename: str):
        ea_calls.append(filename)
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr("loaders.load_channel.load_effective_area_file", _fake_ea)
    monkeypatch.setattr("loaders.load_channel.load_spread_profile_file", _no_spread)

    cfg_path = tmp_path / "ir.cfg"
    _write_cfg(cfg_path, channel_name="IR", effective_area_file="ir.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert isinstance(ch, PhotometryChannel)
    assert ch.channel_name == "IR"
    assert ch.exposure_s == pytest.approx(5.0)
    assert ea_calls == []  # IR skips effective area loading
