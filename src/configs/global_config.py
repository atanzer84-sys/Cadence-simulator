from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    line_core_emission: bool
    interstellar_absorption: bool

    mg2_col: float | None
    mg1_col: float | None
    fe2_col: float | None
    sigmaMg22: float
    sigmaMg21: float
    
    enable_log_r_fallback: bool
    log_r_teff_threshold: float
    log_r_hot_value: float
    log_r_cool_value: float
    
    n_non_science_frames: int
    write_non_science_frames_png: bool
    n_science_frames_per_channel: int
    write_science_frames_png: bool

    cosmic_rays_min: int
    cosmic_rays_max: int
    cosmic_ray_signal_electrons: int
    cosmic_ray_length_min_px: int
    cosmic_ray_length_max_px: int

    # Magnitude cutoff for background star calculation and Gaia fetching (G mag limit).
    magnitude_cutoff: float
    GAIA_USE_ASYNC_JOBS: bool
    test_mode: bool
    produce_Plots: bool

_GLOBAL: GlobalConfig | None = None

DEFAULT_SIGMA_MG22 = 0.257
DEFAULT_SIGMA_MG21 = 0.288

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
    _warn_default_used(raw, "sigmaMg22", DEFAULT_SIGMA_MG22, path=path)
    _warn_default_used(raw, "sigmaMg21", DEFAULT_SIGMA_MG21, path=path)

    cfg = GlobalConfig(
        line_core_emission=_as_bool(raw.get("line_core_emission", 0), key="line_core_emission"),
        interstellar_absorption=_as_bool(raw.get("interstellar_absorption", 0), key="interstellar_absorption"),
        
        mg2_col=_as_optional_float(raw.get("mg2_col", None)),
        mg1_col=_as_optional_float(raw.get("mg1_col", None)),
        fe2_col=_as_optional_float(raw.get("fe2_col", None)),
        sigmaMg22=_as_float(raw.get("sigmaMg22", DEFAULT_SIGMA_MG22), key="sigmaMgIIh"),
        sigmaMg21=_as_float(raw.get("sigmaMg21", DEFAULT_SIGMA_MG21), key="sigmaMgIIk"),
        
        enable_log_r_fallback=_as_bool(raw.get("enable_log_r_fallback", 0), key="enable_log_r_fallback"),
        log_r_teff_threshold=_as_float(raw["log_r_teff_threshold"], key="log_r_teff_threshold"),
        log_r_hot_value=_as_float(raw["log_r_hot_value"], key="log_r_hot_value"),
        log_r_cool_value=_as_float(raw["log_r_cool_value"], key="log_r_cool_value"),
        
        n_non_science_frames=_as_int(raw.get("n_non_science_frames", 0), key="n_non_science_frames"),
        write_non_science_frames_png=_as_bool(raw.get("write_non_science_frames_png", 0), key="write_non_science_frames_png"),
        n_science_frames_per_channel=_as_int(raw.get("n_science_frames_per_channel", 0), key="n_science_frames_per_channel"),
        write_science_frames_png=_as_bool(raw.get("write_science_frames_png", 0), key="write_science_frames_png"),        

        cosmic_rays_min=_as_int(raw.get("cosmic_rays_min", 5), key="cosmic_rays_min"),
        cosmic_rays_max=_as_int(raw.get("cosmic_rays_max", 10), key="cosmic_rays_max"),
        cosmic_ray_signal_electrons=_as_int(raw.get("cosmic_ray_signal_electrons", 720000), key="cosmic_ray_signal_electrons"),
        cosmic_ray_length_min_px=_as_int(raw.get("cosmic_ray_length_min_px", 10), key="cosmic_ray_length_min_px"),
        cosmic_ray_length_max_px=_as_int(raw.get("cosmic_ray_length_max_px", 20), key="cosmic_ray_length_max_px"),

        magnitude_cutoff=_as_float(raw.get("magnitude_cutoff", 20.0), key="magnitude_cutoff"),
        GAIA_USE_ASYNC_JOBS=_as_bool(raw.get("GAIA_USE_ASYNC_JOBS", 1), key="GAIA_USE_ASYNC_JOBS"),

        test_mode=_as_bool(raw.get("test_mode", 0), key="test_mode"),    
        produce_Plots=_as_bool(raw.get("produce_Plots", 0), key="produce_Plots",),    
    )

    _ensure_non_negative(cfg.log_r_teff_threshold, key="log_r_teff_threshold")
    _ensure_non_negative(cfg.n_non_science_frames, key="n_non_science_frames")
    _ensure_non_negative(cfg.n_science_frames_per_channel, key="n_science_frames_per_channel")
    _ensure_non_negative(cfg.cosmic_rays_min, key="cosmic_rays_min")
    _ensure_non_negative(cfg.cosmic_rays_max, key="cosmic_rays_max")
    _ensure_non_negative(cfg.cosmic_ray_length_min_px, key="cosmic_ray_length_min_px")
    _ensure_non_negative(cfg.cosmic_ray_length_max_px, key="cosmic_ray_length_max_px")
    _ensure_min_le_max(cfg.cosmic_rays_min, cfg.cosmic_rays_max, key_min="cosmic_rays_min", key_max="cosmic_rays_max")
    _ensure_min_le_max(cfg.cosmic_ray_length_min_px, cfg.cosmic_ray_length_max_px, key_min="cosmic_ray_length_min_px", key_max="cosmic_ray_length_max_px")
    _ensure_min_le_max(cfg.log_r_hot_value, cfg.log_r_cool_value, key_min="log_r_hot_value", key_max="log_r_cool_value")

    logging.info("Global config loaded: %s", cfg)
    return cfg

def _as_int(value, *, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        logging.error("Invalid int for key '%s': %r", key, value)
        raise ValueError(f"Invalid int for key '{key}': {value!r}") from exc

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

def _as_float(value, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc

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

def _warn_default_used(raw: dict, key: str, default, *, path: Path) -> None:
    if key not in raw:
        logging.warning(
            "%s not provided in %s, using default value %s",
            key,
            path,
            default,
        )

def _ensure_non_negative(value: int, *, key: str) -> int:
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _ensure_min_le_max(min_val: int, max_val: int, *, key_min: str, key_max: str):
    if min_val > max_val:
        raise ValueError(f"{key_min} must be <= {key_max}")