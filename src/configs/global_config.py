from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    line_core_emission: bool
    add_ism_abs: bool
    mg2_col: float | None
    mg1_col: float | None
    fe2_col: float | None
    test_mode: bool


_GLOBAL: GlobalConfig | None = None


def load_global_config(path: Path) -> GlobalConfig:
    """
    Load once and cache. Safe to call multiple times.
    """
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = _read_global_cfg(path)
    return _GLOBAL


def get_global_config() -> GlobalConfig:
    if _GLOBAL is None:
        logging.error(
            "Global config not loaded. "
            "Call load_global_config() once during startup before using it."
        )

        raise RuntimeError(
            "Global config not loaded. Call load_global_config() once in main()."
        )
    return _GLOBAL


def _read_global_cfg(path: Path) -> GlobalConfig:
    logging.info("Reading global config from %s", path)

    raw = _parse_simple_kv(path)

    cfg = GlobalConfig(
        line_core_emission=_as_bool(
            raw.get("line_core_emission", 0),
            key="line_core_emission",
        ),
        add_ism_abs=_as_bool(
            raw.get("add_ism_abs", 0),
            key="add_ism_abs",
        ),
        mg2_col=_as_optional_float(raw.get("mg2_col", None)),
        mg1_col=_as_optional_float(raw.get("mg1_col", None)),
        fe2_col=_as_optional_float(raw.get("fe2_col", None)),
        test_mode=_as_bool(
            raw.get("test_mode", 0),
            key="test_mode",
        ),    
    )
    logging.info("Global config loaded: %s", cfg)
    return cfg


def _as_bool(v: object, *, key: str) -> bool:
    s = str(v).strip().casefold()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False

    logging.error(
        "Invalid boolean value for config key '%s': %r",
        key,
        v,
    )
    raise ValueError(
        f"Invalid boolean value for config key '{key}': {v!r}. "
        "Expected one of: 0, 1, true, false, yes, no."
    )

def _as_optional_float(v: object | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.casefold() == "none":
        return None
    return float(s)

def _parse_simple_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.error("Global config file not found at %s", path)
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