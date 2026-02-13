from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    x_pixels: int
    y_pixels: int

    resolution_factor: float

    dark_current_e_per_s_per_pix: float
    read_noise_e_rms_per_pix: float

    effective_area_file: str


_CHANNEL: ChannelConfig | None = None

def load_channel_config(path: Path) -> ChannelConfig:
    """
    Load once and cache. Safe to call multiple times.
    """
    global _CHANNEL
    if _CHANNEL is None:
        _CHANNEL = _read_channel_cfg(path)
    return _CHANNEL

def get_channel_config() -> ChannelConfig:
    if _CHANNEL is None:
        logging.error(
            "Channel config not loaded. "
            "Call load_channel_config() once during startup before using it."
        )
        raise RuntimeError(
            "Channel config not loaded. Call load_channel_config() once in main()."
        )
    return _CHANNEL

def _read_channel_cfg(path: Path) -> ChannelConfig:
    logging.info("Reading channel config from %s", path)

    raw = _parse_simple_kv(path)

    cfg = ChannelConfig(
        x_pixels=_as_int(raw["x_pixels"], key="x_pixels"),
        y_pixels=_as_int(raw["y_pixels"], key="y_pixels"),
        resolution_factor=_as_float(raw["resolution_factor"], key="resolution_factor"),
        dark_current_e_per_s_per_pix=_as_float( raw["dark_current_e_per_s_per_pix"], key="dark_current_e_per_s_per_pix"),
        read_noise_e_rms_per_pix=_as_float(raw["read_noise_e_rms_per_pix"], key="read_noise_e_rms_per_pix"),
        effective_area_file=raw["effective_area_file"],
    )

    logging.info("Channel config loaded: %s", cfg)
    return cfg

def _as_int(value, *, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        logging.error("Invalid int for key '%s': %r", key, value)
        raise ValueError(f"Invalid int for key '{key}': {value!r}") from exc

def _as_float(value, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc

def _parse_simple_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.error("Channel config file not found at %s", path)
        raise FileNotFoundError(f"Config not found: {path}")

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if "=" not in s:
            continue
        k, v = (p.strip() for p in s.split("=", 1))
        data[k] = v
    return data
