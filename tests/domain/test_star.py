import pytest
from domain.star import Star
from utils.constants import R_SUN_cm, M_SUN_kg


def test_star_from_params_ok_and_unit_conversion():
    # Verifies that Star.from_params computes radius_sun_cm and mass_sun_kg using constants.
    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": 3.0,
    }

    star = Star.from_params(params, required_keys=["name", "radius", "mass"])

    assert star.radius == 2.0
    assert star.mass == 3.0
    assert star.radius_sun_cm == pytest.approx(2.0 * R_SUN_cm)
    assert star.mass_sun_kg == pytest.approx(3.0 * M_SUN_kg)


def test_star_missing_required_key_raises():
    # Verifies that missing required keys cause Star.from_params to raise ValueError.
    params = {
        "radius": 2.0,
    }

    with pytest.raises(ValueError):
        Star.from_params(params, required_keys=["name", "radius", "mass"])


def test_star_empty_required_key_raises():
    # Verifies that empty required key values are treated as missing and raise ValueError.
    params = {
        "name": "   ",
        "radius": 2.0,
    }

    with pytest.raises(ValueError):
        Star.from_params(params, required_keys=["name", "radius", "mass"])


def test_star_missing_effective_temperature_raises():
    # Verifies that when effective_temperature is required but missing, Star.from_params raises ValueError.
    params = {
        "name": "HD 1234",
        "radius": 2.0,
    }

    with pytest.raises(ValueError, match="effective_temperature"):
        Star.from_params(
            params,
            required_keys=["name", "radius", "effective_temperature"],
        )


def test_star_is_frozen():
    # Verifies that Star dataclass is immutable.
    params = {
        "name": "HD 1234",
        "radius": 1.0,
        "mass": 1.0,
    }

    star = Star.from_params(params, required_keys=["name", "radius", "mass"])

    with pytest.raises(Exception):
        star.radius = 5.0
