from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass(frozen=True, slots=True)
class UserConfig:
    target_name: str
    total_observation_length_h: float
    exposure_NUV_s: float
    exposure_VIS_s: float
    exposure_IR_s: float


_USER: UserConfig | None = None


def load_user_config(path: Path) -> UserConfig:
    """
    Load once and cache. Safe to call multiple times.
    """
    global _USER
    if _USER is None:
        _USER = _read_user_cfg(path)
    return _USER

def get_user_config() -> UserConfig:
    if _USER is None:
        logging.error(
            "User config not loaded. "
            "Call load_user_config() once during startup before using it."
        )
        raise RuntimeError(
            "User config not loaded. Call load_user_config() once in main()."
        )
    return _USER


def _read_user_cfg(path: Path) -> UserConfig:
    logging.info("Reading user config from %s", path)
    raw = _parse_simple_kv(path)
    try:
        cfg = UserConfig(
            target_name=_sanitize_target_name(raw["target_name"]),
            total_observation_length_h=_as_float(raw["total_observation_length_h"], key="total_observation_length_h"),
            exposure_NUV_s=_as_float(raw["exposure_NUV_s"], key="exposure_NUV_s"),
            exposure_VIS_s=_as_float(raw["exposure_VIS_s"], key="exposure_VIS_s"),
            exposure_IR_s=_as_float(raw["exposure_IR_s"], key="exposure_IR_s"),
        )
        logging.info("User config loaded: %s", cfg)
        return cfg
    except KeyError as e:
        missing = e.args[0]
        full_path = path.resolve()
        logging.error(
            "Required parameter '%s' is missing in %s",
            missing,
            full_path,
        )
        raise ValueError(
            f"The parameter '{missing}' is missing in the parameter file at {full_path}"
        )


def _sanitize_target_name(v: str) -> str:
    s = str(v).strip()
    # remove quotes if present
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    return s

def _as_float(value, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc

def _parse_simple_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.error("User config file not found at %s", path)
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
