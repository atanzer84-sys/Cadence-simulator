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
