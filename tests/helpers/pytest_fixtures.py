from types import SimpleNamespace

import pytest
from astropy.io import fits

from domain.planet import Planet
from domain.star import Star

__all__ = ["make_channel", "make_photometry_channel", "base_header", "make_star", "make_planet", "make_config", "make_user_cfg"]


@pytest.fixture
def base_header():
    """Base FITS header (matches real pipeline usage)."""
    return fits.Header()


@pytest.fixture
def make_star():
    def _make_star(**overrides):
        base = dict(
            name="HD 1",
            spectral_type=None,
            effective_temperature=None,
            radius=1.0,
            mass=1.0,
            metallicity=None,
            surface_gravity=None,
            right_ascension=None,
            declination=None,
            distance_pc=None,
            v_magnitude=None,
            gaia_magnitude=None,
            log_r=None,
            radius_sun_cm=1.0,
            mass_sun_kg=1.0,
        )
        base.update(overrides)
        return Star(**base)

    return _make_star


@pytest.fixture
def make_planet():
    def _make_planet(**overrides):
        base = dict(
            name="HD 1234 b",
            discoverymethod=None,
            orbital_period=3.5,
            orbital_semi_major_axis=None,
            radius_jupiter=None,
            mass_jupiter=None,
            equilibrium_temperature=None,
            scale_height=None,
        )
        base.update(overrides)
        return Planet(**base)

    return _make_planet


@pytest.fixture
def make_config():
    def _make_config(**overrides):
        base = dict(
            line_core_emission=0,
            interstellar_absorption=0,
            orbit_duration_minutes=100.0,
            orbit_revolutions=1.0,
            readout_gap_s=0.0,
            mg2_col=12.5,
            mg1_col=3,
            fe2_col=0.001,
            sigmaMg22=0.257,
            sigmaMg21=0.288,
            enable_log_r_fallback=1,
            log_r_teff_threshold=5500,
            log_r_hot_value=-4.8,
            log_r_cool_value=-4.2,
            n_calibration_frames=3,
            write_calibration_frame_png=1,
            write_science_frames_png=True,
            write_background_star_png=0,
            background_type="DEFAULT",
            background_file="background_default.txt",
            sky_pixel_area_arcsec2=25.0,
            zod_dist_file="zod_dist.fits",
            zod_spectrum_file="zod_spectrum.txt",
            test_mode=0,
            produce_flux_convolution_plots=1,
        )

        base.update(overrides)
        return SimpleNamespace(**base)

    return _make_config


@pytest.fixture
def make_user_cfg():
    def _make_user_cfg(**overrides):
        base = dict(
            target_name="HD 202772 A",
            total_observation_length_h="20.5",
            exposure_NUV_s="3",
            exposure_VIS_s="4.25",
            exposure_IR_s="10",
        )
        base.update(overrides)
        return "\n".join(f"{k} = {v}" for k, v in base.items())

    return _make_user_cfg
