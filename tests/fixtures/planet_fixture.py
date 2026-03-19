import pytest
from domain.planet import Planet

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