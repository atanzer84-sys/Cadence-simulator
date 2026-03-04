from dataclasses import dataclass
from pathlib import Path
import logging

from configs.config_parsing import parse_simple_kv, as_float

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
    raw = parse_simple_kv(path)
    try:
        cfg = UserConfig(
            target_name=_sanitize_target_name(raw["target_name"]),
            total_observation_length_h=as_float(raw["total_observation_length_h"], key="total_observation_length_h"),
            exposure_NUV_s=as_float(raw["exposure_NUV_s"], key="exposure_NUV_s"),
            exposure_VIS_s=as_float(raw["exposure_VIS_s"], key="exposure_VIS_s"),
            exposure_IR_s=as_float(raw["exposure_IR_s"], key="exposure_IR_s"),
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

