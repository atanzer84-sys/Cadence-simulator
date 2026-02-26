"""
Tests for load_channel_config and load_channels_config.
Migrated from tests/instrument/test_detector.py (calibration/loading tests).
"""
import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path

from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from loaders.load_channel import (
    load_channel_config,
    load_channels_config,
)

# Monkeypatch targets (avoids repeating long strings)
_EA_LOADER = "loaders.load_channel.load_effective_area_file"
_SPREAD_LOADER = "loaders.load_channel.load_spread_profile_file"
_BG_LOADER = "loaders.load_channel.load_background_file"
_REPO_ROOT = "loaders.load_channel.get_repo_root"
# Where file loaders (EA, spread, background) resolve paths:
_REPO_ROOT_FILES = "loaders.load_channel_files.get_repo_root"


@dataclass
class _UserCfgChannels:
    """Minimal user config for load_channels_config (exposure times only)."""
    exposure_NUV_s: float
    exposure_VIS_s: float
    exposure_IR_s: float


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
        "background_file": "",
    }
    defaults.update(kwargs)
    lines = [f"{k} = {v}" for k, v in defaults.items()]
    path.write_text("\n".join(lines), encoding="utf-8")


def _no_spread(_filename: str, _channel_name: str):
    """Return None spread data so tests don't call the real loader."""
    return None, None, None


def _no_background(_filename: str):
    """Return None background data so tests don't call the real loader."""
    return None, None


def _write_ea_file(path: Path, pixel_scale: float = 0.01, rows: int = 3) -> None:
    """Write a minimal effective area file (wavelength, effective_area columns)."""
    lines = [f"# Pixel scale: {pixel_scale}", "Wavelength  EffectiveArea"]
    for i in range(rows):
        wl = 1000.0 + i * 100
        ea = 0.1 + i * 0.1
        lines.append(f"{wl}  {ea}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_spread_file(path: Path, wavelengths: list[float], num_rows: int = 3) -> None:
    """Write a minimal spread profile file (pixels header + dy + weight columns)."""
    wl_str = "  ".join(str(w) for w in wavelengths)
    lines = ["# comment", f"pixels  {wl_str}"]
    for i in range(num_rows):
        dy = float(i)
        weights = "  ".join("0.5" for _ in wavelengths)
        lines.append(f"{dy}  {weights}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# load_channel_config: loader calls and return values
# ----------------------------------------------------------------------


def test_load_channel_config_calls_ea_loader_with_effective_area_file(monkeypatch, tmp_path):
    """load_channel_config calls load_effective_area_file with effective_area_file from cfg."""
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
    """load_channel_config returns SpectroscopyChannel with wavelength, effective_area, pixel_scale from loader."""
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
    """load_channel_config propagates ValueError from load_effective_area_file."""
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
    """load_channel_config raises ValueError when len(wavelength) != x_pixels."""
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
    """load_channel_config stores spread_y_positions, spread_y_weights, spread_y_wavelengths from spread loader."""
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
    """Empty spread_profile_file results in None spread fields."""
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


# ----------------------------------------------------------------------
# load_channel_config: _parse_simple_kv and validation (IR avoids EA load)
# ----------------------------------------------------------------------


def test_load_channel_config_missing_file_raises(tmp_path):
    """Missing config file raises FileNotFoundError."""
    missing = tmp_path / "missing.cfg"
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_channel_config(missing, exposure_s=10.0)


def test_load_channel_config_invalid_int_raises_valueerror(tmp_path):
    """Non-integer x_pixels raises ValueError."""
    cfg_path = tmp_path / "bad_int.cfg"
    _write_cfg(cfg_path, channel_name="IR", x_pixels="not_an_int")

    with pytest.raises(ValueError, match="x_pixels"):
        load_channel_config(cfg_path, exposure_s=10.0)


def test_load_channel_config_invalid_float_raises_valueerror(tmp_path):
    """Non-float resolution_factor raises ValueError."""
    cfg_path = tmp_path / "bad_float.cfg"
    _write_cfg(cfg_path, channel_name="IR", resolution_factor="not_a_float")

    with pytest.raises(ValueError, match="resolution_factor"):
        load_channel_config(cfg_path, exposure_s=10.0)


def test_load_channel_config_ignores_comments(tmp_path):
    """Full-line and inline comments are ignored during parsing."""
    cfg_path = tmp_path / "comments.cfg"
    cfg_path.write_text(
        """
        # detector configuration
        x_pixels = 2048      # detector width
        y_pixels = 1024
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = ir.txt   # calibration file
        channel_name = IR
        dark_current_sigma = 0.001
        mode = 1
        bias_offset = 0.0
        ccd_gain = 1.0
        spread_profile_file =
        spread_half_height_pix = 0
        """,
        encoding="utf-8",
    )

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert ch.x_pixels == 2048
    assert ch.read_noise == pytest.approx(3.2)
    assert ch.channel_name == "IR"


def test_load_channel_config_missing_required_key_raises_keyerror(tmp_path):
    """Missing required key (e.g. mode) raises KeyError."""
    cfg_path = tmp_path / "missing_key.cfg"
    cfg_path.write_text(
        """
        x_pixels = 2048
        y_pixels = 1024
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        channel_name = IR
        dark_current_sigma = 0.001
        """,
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="mode"):
        load_channel_config(cfg_path, exposure_s=10.0)


def test_channel_is_frozen(tmp_path):
    """SpectroscopyChannel and PhotometryChannel are immutable (frozen dataclass)."""
    from dataclasses import FrozenInstanceError

    cfg_path = tmp_path / "ir.cfg"
    _write_cfg(cfg_path, channel_name="IR")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    with pytest.raises(FrozenInstanceError):
        ch.x_pixels = 200


def test_load_channel_config_duplicate_keys_last_wins(tmp_path):
    """When a key appears multiple times, the last occurrence overwrites earlier ones."""
    cfg_path = tmp_path / "channel.cfg"
    cfg_path.write_text(
        """
        effective_area_file = ea.txt
        x_pixels = 1024
        y_pixels = 512
        resolution_factor = 1.0
        dark_noise = 0.0
        read_noise = 3.5
        channel_name = IR
        x_pixels = 2048
        dark_current_sigma = 0.001
        mode = 1
        bias_offset = 0.0
        ccd_gain = 1.0
        spread_profile_file =
        spread_half_height_pix = 0
        """,
        encoding="utf-8",
    )

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.x_pixels == 2048


def test_load_channel_config_spread_half_height_pix_optional_none_uses_zero(monkeypatch, tmp_path):
    """_as_optional_int: spread_half_height_pix = none/empty yields 0."""
    monkeypatch.setattr(_EA_LOADER, lambda _: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    cfg_path = tmp_path / "nuv.cfg"
    _write_cfg(cfg_path, spread_half_height_pix="none")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert ch.spread_half_height_pix == 0


def test_load_channel_config_integration_ea_and_spread_from_real_files(monkeypatch, tmp_path):
    """load_channel_config loads EA and spread from real files when get_repo_root points to tmp_path."""
    monkeypatch.setattr(_REPO_ROOT_FILES, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ea_file(data_dir / "nuv_ea.txt", pixel_scale=0.02, rows=2)
    _write_spread_file(data_dir / "nuv_spread.txt", wavelengths=[10.0, 20.0], num_rows=2)
    # No background file: loader returns (None, None)

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
    """load_channel_config stores background_wavelength and background_flux from load_background_file (not the file path)."""
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
    _write_cfg(cfg_path, effective_area_file="nuv.txt", x_pixels=2, background_file="bg.txt")

    ch = load_channel_config(cfg_path, exposure_s=10.0)

    assert np.allclose(ch.background_wavelength, bg_wl)
    assert np.allclose(ch.background_flux, bg_flux)


# ----------------------------------------------------------------------
# load_channels_config: three-channel orchestration
# ----------------------------------------------------------------------


def test_load_channels_config_calls_load_channel_config_three_times(monkeypatch, tmp_path):
    """load_channels_config calls load_channel_config three times for NUV, VIS, IR."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)

    nuv_cfg = tmp_path / "configs" / "waltzer_nuv.cfg"
    vis_cfg = tmp_path / "configs" / "waltzer_vis.cfg"
    ir_cfg = tmp_path / "configs" / "waltzer_ir.cfg"
    nuv_cfg.parent.mkdir(parents=True, exist_ok=True)

    _write_cfg(nuv_cfg, channel_name="NUV", effective_area_file="nuv.txt", x_pixels=2)
    _write_cfg(vis_cfg, channel_name="VIS", effective_area_file="vis.txt", x_pixels=2)
    _write_cfg(ir_cfg, channel_name="IR", effective_area_file="ir.txt", x_pixels=2)

    def _fake_ea(filename: str):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)
    monkeypatch.setattr(_BG_LOADER, _no_background)

    user_cfg = _UserCfgChannels(exposure_NUV_s=3.0, exposure_VIS_s=4.0, exposure_IR_s=5.0)

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

    monkeypatch.setattr(_EA_LOADER, _fake_ea)
    monkeypatch.setattr(_SPREAD_LOADER, _no_spread)

    cfg_path = tmp_path / "ir.cfg"
    _write_cfg(cfg_path, channel_name="IR", effective_area_file="ir.txt", x_pixels=2)

    ch = load_channel_config(cfg_path, exposure_s=5.0)

    assert isinstance(ch, PhotometryChannel)
    assert ch.channel_name == "IR"
    assert ch.exposure_s == pytest.approx(5.0)
    assert ea_calls == []  # IR skips effective area loading
