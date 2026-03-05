"""
Shared helpers for load_channel tests.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import configs.global_config as gc
from tests.helpers.global_config_factory import make_global_cfg

from loaders.load_channel import (
    load_channel_config as _load_channel_config,
    load_channels_config as _load_channels_config,
    _load_background_from_global_cfg,
)

# Monkeypatch targets (avoids repeating long strings)
_EA_LOADER = "loaders.load_channel.load_effective_area_file"
_SPREAD_LOADER = "loaders.load_channel.load_spread_profile_file_spectroscopy"
_BG_LOADER = "loaders.load_channel.load_background_file"
_ZOD_DIST_LOADER = "loaders.load_channel.load_zod_dist_file"
_ZOD_SPECTRUM_LOADER = "loaders.load_channel.load_zod_spectrum_file"
_PSF_LOADER = "loaders.load_channel.load_psf_image_file"
_REPO_ROOT = "loaders.load_channel.get_repo_root"
_REPO_ROOT_FILES_COMMON = "loaders.load_channel_files_common.get_repo_root"
_REPO_ROOT_FILES_SPEC = "loaders.load_channel_files_spectroscopy.get_repo_root"
_REPO_ROOT_FILES_PHOT = "loaders.load_channel_files_photometry.get_repo_root"

# Single shared config template for all tests.
_SHARED_CFG = {
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
    # NIR-only fields kept here so tests can switch channel_name without
    # repeating another config template.
    "aperture_pix": 4.0,
    "psf_file": "nir_psf.txt",
    "source_position_x_arcsec": 0.0,
    "source_position_y_arcsec": 0.0,
}
_CTX = SimpleNamespace(output_dir=Path("."), target_name="TEST")


def set_test_global_cfg(**overrides) -> None:
    """Set test GlobalConfig used by loaders.load_channel.get_global_config()."""
    gc._GLOBAL = make_global_cfg(**overrides)


def load_channel_config(path: Path, exposure_s: float):
    """Compatibility wrapper for tests after ctx-aware loader refactor."""
    background = _load_background_from_global_cfg()
    return _load_channel_config(path, exposure_s, _CTX, background)


def load_channels_config(user_cfg):
    """Compatibility wrapper for tests after ctx-aware loader refactor."""
    return _load_channels_config(user_cfg, _CTX)


@dataclass
class _UserCfgChannels:
    """Minimal user config for load_channels_config (exposure times only)."""

    exposure_NUV_s: float
    exposure_VIS_s: float
    exposure_IR_s: float


def write_cfg(path: Path, remove_keys: set[str] | None = None, **kwargs) -> None:
    """Write a test channel cfg from one shared template + overrides."""
    cfg = dict(_SHARED_CFG)
    cfg.update(kwargs)
    if remove_keys:
        for key in remove_keys:
            cfg.pop(key, None)
    lines = [f"{k} = {v}" for k, v in cfg.items()]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cfg_with_comments(path: Path, **kwargs) -> None:
    """Write config from shared template while preserving inline comment parsing behavior."""
    cfg = dict(_SHARED_CFG)
    cfg.update(kwargs)
    lines = [
        "# detector configuration",
        f"x_pixels = {cfg['x_pixels']}      # detector width",
        f"y_pixels = {cfg['y_pixels']}",
        f"resolution_factor = {cfg['resolution_factor']}",
        f"dark_noise = {cfg['dark_noise']}",
        f"read_noise = {cfg['read_noise']}",
        f"effective_area_file = {cfg['effective_area_file']}   # calibration file",
        f"channel_name = {cfg['channel_name']}",
        f"dark_current_sigma = {cfg['dark_current_sigma']}",
        f"mode = {cfg['mode']}",
        f"bias_offset = {cfg['bias_offset']}",
        f"ccd_gain = {cfg['ccd_gain']}",
        f"psf_file = {cfg['psf_file']}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def no_spread(_filename: str, _channel_name: str):
    """Return None spread data so tests don't call the real loader."""
    return None, None, None


def no_background(_filename: str):
    """Return None background data so tests don't call the real loader."""
    return None, None


def fake_psf_image_loader(_filename: str, _channel_name: str, _ctx):
    """Return a tiny PSF image + center for NIR tests."""
    return np.array([[1.0]]), 0, 0


def write_ea_file(path: Path, pixel_scale: float = 0.01, rows: int = 3) -> None:
    """Write a minimal effective area file (wavelength, effective_area columns)."""
    lines = [f"# Pixel scale: {pixel_scale}", "Wavelength  EffectiveArea"]
    for i in range(rows):
        wl = 1000.0 + i * 100
        ea = 0.1 + i * 0.1
        lines.append(f"{wl}  {ea}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spread_file(path: Path, wavelengths: list[float], num_rows: int = 3) -> None:
    """Write a minimal spread profile file (pixels header + dy + weight columns)."""
    wl_str = "  ".join(str(w) for w in wavelengths)
    lines = ["# comment", f"pixels  {wl_str}"]
    for i in range(num_rows):
        dy = float(i)
        weights = "  ".join("0.5" for _ in wavelengths)
        lines.append(f"{dy}  {weights}")
    path.write_text("\n".join(lines), encoding="utf-8")
