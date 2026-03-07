"""Shared config file parsing utilities for key=value format."""

from pathlib import Path
import logging


def parse_simple_kv(path: Path) -> dict[str, str]:
    """Parse a simple key=value config file. Skips empty lines and # comments."""
    if not path.exists():
        logging.error("Config file not found at %s", path)
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


def as_int(value: object, *, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        logging.error("Invalid int for key '%s': %r", key, value)
        raise ValueError(f"Invalid int for key '{key}': {value!r}") from exc


def as_float(value: object, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc


def as_bool(v: object, *, key: str) -> bool:
    s = str(v).strip().casefold()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False

    logging.error("Invalid boolean value for config key '%s': %r", key, v)
    raise ValueError(
        f"Invalid boolean value for config key '{key}': {v!r}. "
        "Expected one of: 0, 1, true, false, yes, no."
    )


def as_optional_int(v: object | None) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.casefold() == "none":
        return None
    try:
        return int(s)
    except Exception as exc:
        logging.error("Invalid int value: %r", v)
        raise ValueError(f"Invalid int value: {v!r}") from exc


def as_optional_float(v: object | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.casefold() == "none":
        return None
    return float(s)


def as_optional_str(v: object | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.casefold() == "none":
        return None
    return s


def as_optional_lower_str(v: object | None) -> str | None:
    s = as_optional_str(v)
    if s is None:
        return None
    return s.casefold()
