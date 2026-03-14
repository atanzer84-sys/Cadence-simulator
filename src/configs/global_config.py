from dataclasses import dataclass
from pathlib import Path
import logging

from configs.config_parsing import (parse_simple_kv, as_bool, as_optional_float, as_float, as_int, as_optional_lower_str, as_optional_str)


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    run_vis: bool
    run_nuv: bool
    run_nir: bool
    line_core_emission: bool
    interstellar_absorption: bool
    orbit_duration_minutes: float
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
    
    n_calibration_frames: int
    write_calibration_frames_png: bool
    write_science_frame_component_png: bool
    invert_calibration_science_frame_component: bool
    invert_science_frames : bool
    write_science_frames_png: bool
    write_background_star_footprint_on_science_frame: bool

    cosmic_rays_min: int
    cosmic_rays_max: int
    cosmic_ray_signal_electrons: int
    cosmic_ray_length_min_px: int
    cosmic_ray_length_max_px: int

    magnitude_cutoff: float
    GAIA_USE_ASYNC_JOBS: bool
    gaia_conesearch_radius_arcsec: float

    background_type: str | None
    background_file: str | None
    sky_pixel_area_arcsec2: float | None
    zod_dist_file: str | None
    zod_spectrum_file: str | None   

    write_intermediate_arrays: bool
    produce_flux_convolution_plots: bool
    produce_target_background_star_noise_vs_counts_plot: bool

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
        logging.error("Global config not loaded. Call load_global_config() once during startup before using it.")

        raise RuntimeError(
            "Global config not loaded. Call load_global_config() once in main()."
        )
    return _GLOBAL

def _read_global_cfg(path: Path) -> GlobalConfig:
    raw = parse_simple_kv(path)
    _warn_default_used(raw, "sigmaMg22", DEFAULT_SIGMA_MG22, path=path)
    _warn_default_used(raw, "sigmaMg21", DEFAULT_SIGMA_MG21, path=path)

    run_vis = as_bool(raw.get("run_vis", 1), key="run_vis")
    run_nuv = as_bool(raw.get("run_nuv", 1), key="run_nuv")
    run_nir = as_bool(raw.get("run_nir", 1), key="run_nir")
    line_core_emission = as_bool(raw.get("line_core_emission", 0), key="line_core_emission")
    interstellar_absorption = as_bool(raw.get("interstellar_absorption", 0), key="interstellar_absorption")
    orbit_duration_minutes = as_float(raw.get("orbit_duration_minutes", 100.0), key="orbit_duration_minutes")
    orbit_revolutions = as_float(raw.get("orbit_revolutions", 1.0), key="orbit_revolutions")
    orbit_total_duration_s = _compute_total_simulation_time_s(orbit_duration_minutes, orbit_revolutions)
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


    n_calibration_frames = as_int(raw.get("n_calibration_frames", 0), key="n_calibration_frames")
    write_calibration_frames_png = as_bool(raw.get("write_calibration_frames_png", 0), key="write_calibration_frames_png")
    write_science_frame_component_png = as_bool(raw.get("write_science_frame_component_png", 0), key="write_science_frame_component_png")
    invert_calibration_science_frame_component = as_bool(raw.get("invert_calibration_science_frame_component", 0), key="invert_calibration_science_frame_component")
    invert_science_frames = as_bool(raw.get("invert_science_frames", 0), key="invert_science_frames")
    write_science_frames_png = as_bool(raw.get("write_science_frames_png", 0), key="write_science_frames_png")
    write_background_star_footprint_on_science_frame = as_bool(raw.get("write_background_star_footprint_on_science_frame", 0), key="write_background_star_footprint_on_science_frame")

    cosmic_rays_min = as_int(raw.get("cosmic_rays_min", 5), key="cosmic_rays_min")
    cosmic_rays_max = as_int(raw.get("cosmic_rays_max", 10), key="cosmic_rays_max")
    cosmic_ray_signal_electrons = as_int(raw.get("cosmic_ray_signal_electrons", 72000), key="cosmic_ray_signal_electrons")
    cosmic_ray_length_min_px = as_int(raw.get("cosmic_ray_length_min_px", 10), key="cosmic_ray_length_min_px")
    cosmic_ray_length_max_px = as_int(raw.get("cosmic_ray_length_max_px", 20), key="cosmic_ray_length_max_px")

    magnitude_cutoff = as_float(raw.get("magnitude_cutoff", 20.0), key="magnitude_cutoff")
    GAIA_USE_ASYNC_JOBS = as_bool(raw.get("GAIA_USE_ASYNC_JOBS", 1), key="GAIA_USE_ASYNC_JOBS")
    gaia_conesearch_radius_arcsec = as_float(raw.get("gaia_conesearch_radius_arcsec", 500.0), key="gaia_conesearch_radius_arcsec")

    background_type = as_optional_lower_str(raw.get("background_type", ""))
    background_file = as_optional_str(raw.get("background_file", ""))
    sky_pixel_area_arcsec2 = as_optional_float(raw.get("sky_pixel_area_arcsec2", None))
    zod_dist_file = as_optional_str(raw.get("zod_dist_file", ""))
    zod_spectrum_file = as_optional_str(raw.get("zod_spectrum_file", ""))

    write_intermediate_arrays = as_bool(raw.get("write_intermediate_arrays", 0), key="write_intermediate_arrays")
    produce_flux_convolution_plots = as_bool(raw.get("produce_flux_convolution_plots", raw.get("produce_flux_convolution_plots", 0)), key="produce_flux_convolution_plots")
    produce_target_background_star_noise_vs_counts_plot = as_bool(raw.get("produce_target_background_star_noise_vs_counts_plot", 0), key="produce_target_background_star_noise_vs_counts_plot")

    _ensure_at_least_one_channel_enabled(run_vis, run_nuv, run_nir)
    _ensure_non_negative(orbit_duration_minutes, key="orbit_duration_minutes")
    _ensure_non_negative(orbit_revolutions, key="orbit_revolutions")
    _ensure_non_negative(readout_gap_s, key="readout_gap_s")
    _ensure_non_negative(log_r_teff_threshold, key="log_r_teff_threshold")
    _ensure_non_negative(n_calibration_frames, key="n_calibration_frames")
    _ensure_non_negative(cosmic_rays_min, key="cosmic_rays_min")
    _ensure_non_negative(cosmic_rays_max, key="cosmic_rays_max")
    _ensure_non_negative(cosmic_ray_length_min_px, key="cosmic_ray_length_min_px")
    _ensure_non_negative(cosmic_ray_length_max_px, key="cosmic_ray_length_max_px")
    _ensure_non_negative(gaia_conesearch_radius_arcsec, key="gaia_conesearch_radius_arcsec")
    _ensure_non_negative(magnitude_cutoff, key="magnitude_cutoff")
    _ensure_min_le_max(cosmic_rays_min, cosmic_rays_max, key_min="cosmic_rays_min", key_max="cosmic_rays_max")
    _ensure_min_le_max(cosmic_ray_length_min_px, cosmic_ray_length_max_px, key_min="cosmic_ray_length_min_px", key_max="cosmic_ray_length_max_px")
    _ensure_min_le_max(log_r_hot_value, log_r_cool_value, key_min="log_r_hot_value", key_max="log_r_cool_value")

    cfg = GlobalConfig(
        run_vis=run_vis,
        run_nuv=run_nuv,
        run_nir=run_nir,
        line_core_emission=line_core_emission,
        interstellar_absorption=interstellar_absorption,
        orbit_duration_minutes=orbit_duration_minutes,
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
        n_calibration_frames=n_calibration_frames,
        invert_calibration_science_frame_component=invert_calibration_science_frame_component,
        invert_science_frames=invert_science_frames,
        write_calibration_frames_png=write_calibration_frames_png,
        write_science_frame_component_png=write_science_frame_component_png,
        write_science_frames_png=write_science_frames_png,
        write_background_star_footprint_on_science_frame=write_background_star_footprint_on_science_frame,
        cosmic_rays_min=cosmic_rays_min,
        cosmic_rays_max=cosmic_rays_max,
        cosmic_ray_signal_electrons=cosmic_ray_signal_electrons,
        cosmic_ray_length_min_px=cosmic_ray_length_min_px,
        cosmic_ray_length_max_px=cosmic_ray_length_max_px,
        magnitude_cutoff=magnitude_cutoff,
        GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS,
        gaia_conesearch_radius_arcsec=gaia_conesearch_radius_arcsec,
        background_type=background_type,
        background_file=background_file,
        sky_pixel_area_arcsec2=sky_pixel_area_arcsec2,
        zod_dist_file=zod_dist_file,
        zod_spectrum_file=zod_spectrum_file,
        write_intermediate_arrays=write_intermediate_arrays,
        produce_flux_convolution_plots=produce_flux_convolution_plots,
        produce_target_background_star_noise_vs_counts_plot=produce_target_background_star_noise_vs_counts_plot,
    )

    logging.info("Global config loaded: %s", cfg)
    return cfg


def _ensure_at_least_one_channel_enabled(run_vis: bool, run_nuv: bool, run_nir: bool) -> None:
    if not (run_vis or run_nuv or run_nir):
        raise ValueError("At least one channel must be enabled in global.cfg: run_vis, run_nuv, or run_nir")

def _warn_default_used(raw: dict, key: str, default, *, path: Path) -> None:
    if key not in raw:
        logging.warning("%s not provided in %s, using default value %s", key, path, default)

def _ensure_non_negative(value: float | int, *, key: str) -> float | int:
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _ensure_min_le_max(min_val: float | int, max_val: float | int, *, key_min: str, key_max: str) -> None:
    if min_val > max_val:
        raise ValueError(f"{key_min} must be <= {key_max}")

def _compute_total_simulation_time_s(orbit_duration_minutes: float, orbit_revolutions: float) -> float:
    return float(orbit_duration_minutes) * 60.0 * float(orbit_revolutions)