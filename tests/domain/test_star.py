import pytest
from dataclasses import FrozenInstanceError

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


# Tests: Star dataclass
# Behavior: modifying a frozen dataclass raises FrozenInstanceError
def test_star_is_frozen():
    params = {
        "name": "HD 1234",
        "radius": 1.0,
        "mass": 1.0,
    }

    star = Star.from_params(params, required_keys=["name", "radius", "mass"])

    with pytest.raises(FrozenInstanceError):
        star.radius = 5.0


# Tests: Star.from_params
# Behavior: required_keys empty → no validation → Star created successfully
def test_star_required_keys_empty():
    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": 3.0,
    }

    star = Star.from_params(params, required_keys=[])

    assert star.name == "HD 1234"
    assert star.radius == 2.0
    assert star.mass == 3.0


# Tests: Star.from_params
# Behavior: required key present but is_missing() returns True → raises ValueError
def test_star_required_key_present_but_is_missing(monkeypatch):
    monkeypatch.setattr("domain.star.is_missing", lambda v: True)

    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": 3.0,
    }

    with pytest.raises(ValueError) as exc:
        Star.from_params(params, required_keys=["name", "radius", "mass"])

    msg = str(exc.value)
    assert "radius" in msg or "mass" in msg or "name" in msg
    assert "missing" in msg.lower()


# Tests: Star.from_params
# Behavior: log_output=False → no print/logging → Star created successfully
def test_star_log_output_false(capsys, caplog):
    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": 3.0,
    }

    caplog.clear()
    star = Star.from_params(params, required_keys=["name", "radius", "mass"], log_output=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert not caplog.records

    assert star.name == "HD 1234"


# Tests: Star.from_params
# Behavior: radius=None triggers second-stage radius check → raises ValueError
def test_star_radius_none_raises():
    params = {
        "name": "HD 1234",
        "radius": None,
        "mass": 3.0,
    }

    with pytest.raises(ValueError, match="radius"):
        Star.from_params(params, required_keys=["name", "radius", "mass"])


def test_star_optional_fields_are_passed_through():
    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": 3.0,
        "spectral_type": "G2V",
        "effective_temperature": 5800.0,
        "metallicity": -0.1,
        "surface_gravity": 4.4,
        "right_ascension": 123.0,
        "declination": -45.0,
        "distance": 10.0,
        "v_magnitude": 5.5,
        "gaia_magnitude": 5.1,
        "log_r": 0.2,
    }

    star = Star.from_params(
        params,
        required_keys=["name", "radius", "mass"],
        log_output=False,
    )

    assert star.spectral_type == "G2V"
    assert star.effective_temperature == 5800.0
    assert star.metallicity == -0.1
    assert star.surface_gravity == 4.4
    assert star.right_ascension == 123.0
    assert star.declination == -45.0
    assert star.distance_pc == 10.0
    assert star.v_magnitude == 5.5
    assert star.gaia_magnitude == 5.1
    assert star.log_r == 0.2


def test_star_mass_none_sets_mass_sun_kg_to_none():
    params = {
        "name": "HD 1234",
        "radius": 2.0,
        "mass": None,
    }

    star = Star.from_params(
        params,
        required_keys=["name", "radius"],
        log_output=False,
    )

    assert star.mass is None
    assert star.mass_sun_kg is None


def test_star_radius_none_raises_even_if_radius_not_in_required_keys():
    params = {
        "name": "HD 1234",
        "radius": None,
        "mass": 3.0,
    }

    with pytest.raises(ValueError, match="radius"):
        Star.from_params(
            params,
            required_keys=["name", "mass"],
            log_output=False,
        )