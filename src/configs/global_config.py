from dataclasses import dataclass
from pathlib import Path
import logging

from configs.config_parsing import (parse_simple_kv, as_bool, as_optional_float, as_float, as_int, as_optional_lower_str, as_optional_str)


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    line_core_emission: bool
    interstellar_absorption: bool
    orbit_duration_min: float
    orbit_revolutions: float
    orbit_total_duration_s: float
    readout_gap_s: float

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
    write_science_frames_png: bool

    cosmic_rays_min: int
    cosmic_rays_max: int
    cosmic_ray_signal_electrons: int
    cosmic_ray_length_min_px: int
    cosmic_ray_length_max_px: int

    # Magnitude cutoff for background star calculation and Gaia fetching (G mag limit).
    magnitude_cutoff: float
    GAIA_USE_ASYNC_JOBS: bool

    background_type: str | None
    background_file: str | None
    sky_pixel_area_arcsec2: float | None
    zod_dist_file: str | None
    zod_spectrum_file: str | None   

    test_mode: bool
    produce_plots: bool

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

    raw = parse_simple_kv(path)
    _warn_default_used(raw, "sigmaMg22", DEFAULT_SIGMA_MG22, path=path)
    _warn_default_used(raw, "sigmaMg21", DEFAULT_SIGMA_MG21, path=path)

    line_core_emission = as_bool(raw.get("line_core_emission", 0), key="line_core_emission")
    interstellar_absorption = as_bool(raw.get("interstellar_absorption", 0), key="interstellar_absorption")
    orbit_duration_min = as_float(raw.get("orbit_duration_min", 100.0), key="orbit_duration_min")
    orbit_revolutions = as_float(raw.get("orbit_revolutions", 1.0), key="orbit_revolutions")
    orbit_total_duration_s = _compute_total_simulation_time_s(orbit_duration_min, orbit_revolutions)
    readout_gap_s = as_float(raw.get("readout_gap_s", 0.0), key="readout_gap_s")

    mg2_col = as_optional_float(raw.get("mg2_col", None))
    mg1_col = as_optional_float(raw.get("mg1_col", None))
    fe2_col = as_optional_float(raw.get("fe2_col", None))
    sigmaMg22 = as_float(raw.get("sigmaMg22", DEFAULT_SIGMA_MG22), key="sigmaMgIIh")
    sigmaMg21 = as_float(raw.get("sigmaMg21", DEFAULT_SIGMA_MG21), key="sigmaMgIIk")

    enable_log_r_fallback = as_bool(raw.get("enable_log_r_fallback", 0), key="enable_log_r_fallback")
    log_r_teff_threshold = as_float(raw["log_r_teff_threshold"], key="log_r_teff_threshold")
    log_r_hot_value = as_float(raw["log_r_hot_value"], key="log_r_hot_value")
    log_r_cool_value = as_float(raw["log_r_cool_value"], key="log_r_cool_value")

    n_non_science_frames = as_int(raw.get("n_non_science_frames", 0), key="n_non_science_frames")
    write_non_science_frames_png = as_bool(raw.get("write_non_science_frames_png", 0), key="write_non_science_frames_png")
    write_science_frames_png = as_bool(raw.get("write_science_frames_png", 0), key="write_science_frames_png")

    cosmic_rays_min = as_int(raw.get("cosmic_rays_min", 5), key="cosmic_rays_min")
    cosmic_rays_max = as_int(raw.get("cosmic_rays_max", 10), key="cosmic_rays_max")
    cosmic_ray_signal_electrons = as_int(raw.get("cosmic_ray_signal_electrons", 72000), key="cosmic_ray_signal_electrons")
    cosmic_ray_length_min_px = as_int(raw.get("cosmic_ray_length_min_px", 10), key="cosmic_ray_length_min_px")
    cosmic_ray_length_max_px = as_int(raw.get("cosmic_ray_length_max_px", 20), key="cosmic_ray_length_max_px")

    magnitude_cutoff = as_float(raw.get("magnitude_cutoff", 20.0), key="magnitude_cutoff")
    GAIA_USE_ASYNC_JOBS = as_bool(raw.get("GAIA_USE_ASYNC_JOBS", 1), key="GAIA_USE_ASYNC_JOBS")

    background_type = as_optional_lower_str(raw.get("background_type", ""))
    background_file = as_optional_str(raw.get("background_file", ""))
    sky_pixel_area_arcsec2 = as_optional_float(raw.get("sky_pixel_area_arcsec2", None))
    zod_dist_file = as_optional_str(raw.get("zod_dist_file", ""))
    zod_spectrum_file = as_optional_str(raw.get("zod_spectrum_file", ""))

    test_mode = as_bool(raw.get("test_mode", 0), key="test_mode")
    produce_plots = as_bool(raw.get("produce_plots", raw.get("produce_Plots", 0)), key="produce_plots")

    _ensure_non_negative(orbit_duration_min, key="orbit_duration_min")
    _ensure_non_negative(orbit_revolutions, key="orbit_revolutions")
    _ensure_non_negative(readout_gap_s, key="readout_gap_s")
    _ensure_non_negative(log_r_teff_threshold, key="log_r_teff_threshold")
    _ensure_non_negative(n_non_science_frames, key="n_non_science_frames")
    _ensure_non_negative(cosmic_rays_min, key="cosmic_rays_min")
    _ensure_non_negative(cosmic_rays_max, key="cosmic_rays_max")
    _ensure_non_negative(cosmic_ray_length_min_px, key="cosmic_ray_length_min_px")
    _ensure_non_negative(cosmic_ray_length_max_px, key="cosmic_ray_length_max_px")
    _ensure_min_le_max(cosmic_rays_min, cosmic_rays_max, key_min="cosmic_rays_min", key_max="cosmic_rays_max")
    _ensure_min_le_max(cosmic_ray_length_min_px, cosmic_ray_length_max_px, key_min="cosmic_ray_length_min_px", key_max="cosmic_ray_length_max_px")
    _ensure_min_le_max(log_r_hot_value, log_r_cool_value, key_min="log_r_hot_value", key_max="log_r_cool_value")

    cfg = GlobalConfig(
        line_core_emission=line_core_emission,
        interstellar_absorption=interstellar_absorption,
        orbit_duration_min=orbit_duration_min,
        orbit_revolutions=orbit_revolutions,
        orbit_total_duration_s=orbit_total_duration_s,
        readout_gap_s=readout_gap_s,
        mg2_col=mg2_col,
        mg1_col=mg1_col,
        fe2_col=fe2_col,
        sigmaMg22=sigmaMg22,
        sigmaMg21=sigmaMg21,
        enable_log_r_fallback=enable_log_r_fallback,
        log_r_teff_threshold=log_r_teff_threshold,
        log_r_hot_value=log_r_hot_value,
        log_r_cool_value=log_r_cool_value,
        n_non_science_frames=n_non_science_frames,
        write_non_science_frames_png=write_non_science_frames_png,
        write_science_frames_png=write_science_frames_png,
        cosmic_rays_min=cosmic_rays_min,
        cosmic_rays_max=cosmic_rays_max,
        cosmic_ray_signal_electrons=cosmic_ray_signal_electrons,
        cosmic_ray_length_min_px=cosmic_ray_length_min_px,
        cosmic_ray_length_max_px=cosmic_ray_length_max_px,
        magnitude_cutoff=magnitude_cutoff,
        GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS,
        background_type=background_type,
        background_file=background_file,
        sky_pixel_area_arcsec2=sky_pixel_area_arcsec2,
        zod_dist_file=zod_dist_file,
        zod_spectrum_file=zod_spectrum_file,
        test_mode=test_mode,
        produce_plots=produce_plots,
    )

    logging.info("Global config loaded: %s", cfg)
    return cfg

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

def _compute_total_simulation_time_s(orbit_duration_min: float, orbit_revolutions: float) -> float:
    return float(orbit_duration_min) * 60.0 * float(orbit_revolutions)