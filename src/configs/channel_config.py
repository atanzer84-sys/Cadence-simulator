from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    x_pixels: int
    y_pixels: int
    resolution_factor: float
    dark_noise: float
    dark_current_sigma: float
    read_noise: float
    effective_area_file: str

    bias_offset: float = 0.0
    channel_name: str = ""
    ccd_gain: float = 1.0   # electrons per ADU
    mode: int = 1   

    spread_profile_file: str = ""
    spread_half_height_pix: int = 0

    source_file: str = ""


def load_channel_config(path: Path) -> ChannelConfig:
    logging.info("Reading channel config from %s", path)

    raw = _parse_simple_kv(path)

    cfg = ChannelConfig(
        x_pixels=_as_int(raw["x_pixels"], key="x_pixels"),
        y_pixels=_as_int(raw["y_pixels"], key="y_pixels"),
        resolution_factor=_as_float(raw["resolution_factor"], key="resolution_factor"),
        dark_noise=_as_float(raw["dark_noise"], key="dark_noise"),
        dark_current_sigma=_as_float(raw["dark_current_sigma"], key="dark_current_sigma"),
        read_noise=_as_float(raw["read_noise"], key="read_noise"),
        effective_area_file=raw["effective_area_file"],
        bias_offset=_as_float(raw.get("bias_offset", 0.0), key="bias_offset"),
        channel_name=str(raw["channel_name"]).strip(),
        ccd_gain=_as_float(raw.get("ccd_gain", 1.0), key="ccd_gain"),
        mode=_as_int(raw["mode"], key="mode"),
        spread_profile_file=str(raw.get("spread_profile_file", "")).strip(),
        spread_half_height_pix=_as_optional_int(raw.get("spread_half_height_pix", None)) or 0,
        source_file=str(path),
    )

    logging.info("Channel config loaded: %s", cfg)
    return cfg

def _as_optional_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.casefold() == "none":
        return None
    try:
        return int(s)
    except Exception as exc:
        logging.error("Invalid int value: %r", value)
        raise ValueError(f"Invalid int value: {value!r}") from exc

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
