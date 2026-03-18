"""Pytest conftest: add repo src to path so imports work when run from repo root."""
import sys
from pathlib import Path

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
