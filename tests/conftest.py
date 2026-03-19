"""Pytest conftest: add repo src to path so imports work when run from repo root."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


import pytest
from domain.star import Star

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
