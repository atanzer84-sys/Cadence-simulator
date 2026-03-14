"""Shared GlobalConfig factory for tests."""

from configs.global_config import GlobalConfig

# Keep this in sync with required GlobalConfig fields.
BASE_GLOBAL_CFG = {
    # Channel enable flags (new required fields)
    "run_vis": True,
    "run_nuv": True,
    "run_nir": True,
    # Existing defaults
    "line_core_emission": False,
    "interstellar_absorption": False,
    "orbit_duration_minutes": 100.0,
    "orbit_revolutions": 1.0,
    "orbit_total_duration_s": 6000.0,  # 100 min * 60 s/min * 1 rev
    "readout_gap_s": 0.0,
    "mg2_col": None,
    "mg1_col": None,
    "fe2_col": None,
    "sigmaMg22": 0.257,
    "sigmaMg21": 0.288,
    "enable_log_r_fallback": False,
    "log_r_teff_threshold": 0.0,
    "log_r_hot_value": 0.0,
    "log_r_cool_value": 0.0,
    "n_calibration_frames": 0,
    "invert_calibration_science_frame_component": False,
    "invert_science_frames": False,
    "write_calibration_frame_png": False,
    "write_science_frame_component_png": False,
    "write_science_frames_png": False,
    "write_background_star_footprint_on_science_frame": False,
    "science_frame_png_crop_spectrum_region": False,
    "cosmic_rays_min": 0,
    "cosmic_rays_max": 0,
    "cosmic_ray_signal_electrons": 72000,
    "cosmic_ray_length_min_px": 1,
    "cosmic_ray_length_max_px": 1,
    "magnitude_cutoff": 20.0,
    "GAIA_USE_ASYNC_JOBS": 0,
    "background_type": None,
    "background_file": None,
    "sky_pixel_area_arcsec2": None,
    "zod_dist_file": None,
    "zod_spectrum_file": None,
    "write_intermediate_arrays": False,
    "produce_flux_convolution_plots": False,
    "produce_target_background_star_noise_vs_counts_plot": False,
    "gaia_conesearch_radius_arcsec": 500.0,
}


def make_global_cfg(**overrides) -> GlobalConfig:
    """Build a GlobalConfig from shared defaults with per-test overrides."""
    params = dict(BASE_GLOBAL_CFG)
    params.update(overrides)
    return GlobalConfig(**params)
