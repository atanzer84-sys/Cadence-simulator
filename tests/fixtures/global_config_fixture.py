import pytest
from configs.global_config import GlobalConfig

@pytest.fixture
def make_global_config():
    def _make_global_config(**overrides):
        base = dict(
            run_vis=True,
            run_nuv=True,
            run_nir=True,

            line_core_emission=False,
            interstellar_absorption=False,

            orbit_duration_minutes=100.0,
            orbit_revolutions=1.0,
            orbit_total_duration_s=6000.0,
            readout_gap_s=20.0,

            mg2_col=None,
            mg1_col=None,
            fe2_col=None,
            sigmaMg22=0.257,
            sigmaMg21=0.288,

            enable_log_r_fallback=True,
            log_r_teff_threshold=6800,
            log_r_hot_value=-7,
            log_r_cool_value=-5,

            n_calibration_frames=0,
            write_calibration_frame_png=False,
            write_science_frame_component_png=False,
            invert_calibration_science_frame_component=False,
            invert_science_frames=False,
            write_science_frames_png=False,
            science_frame_png_crop_spectrum_region=False,
            write_background_star_footprint_on_science_frame=False,

            cosmic_rays_min=0,
            cosmic_rays_max=0,
            cosmic_ray_signal_electrons=72000,
            cosmic_ray_length_min_px=10,
            cosmic_ray_length_max_px=20,

            magnitude_cutoff=20.0,
            GAIA_USE_ASYNC_JOBS=False,
            gaia_conesearch_radius_arcsec=500.0,

            background_type=None,
            background_file=None,
            sky_pixel_area_arcsec2=None,
            zod_dist_file=None,
            zod_spectrum_file=None,

            write_intermediate_arrays=False,
            produce_flux_convolution_plots=False,
        )

        base.update(overrides)
        return GlobalConfig(**base)

    return _make_global_config