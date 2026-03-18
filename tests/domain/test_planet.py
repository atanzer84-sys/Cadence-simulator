import pytest
from domain.planet import Planet


def test_planet_from_params_ok():
    # Verifies that Planet.from_params creates a Planet when all required keys are present.
    params = {
        "name": "HD 1234 b",
        "orbital_period": 3.5,
    }

    planet = Planet.from_params(params, required_keys=["name", "orbital_period"])

    assert planet.name == "HD 1234 b"
    assert planet.orbital_period == 3.5


def test_planet_missing_required_key_raises():
    # Verifies that missing required keys cause Planet.from_params to raise ValueError.
    params = {
        "name": "HD 1234 b",
    }

    with pytest.raises(ValueError):
        Planet.from_params(params, required_keys=["name", "orbital_period"])


def test_planet_empty_string_required_key_raises():
    # Verifies that empty strings in required keys are treated as missing and raise ValueError.
    params = {
        "name": "",
        "orbital_period": 3.5,
    }

    with pytest.raises(ValueError):
        Planet.from_params(params, required_keys=["name", "orbital_period"])


def test_planet_is_frozen():
    # Verifies that Planet dataclass is immutable.
    params = {
        "name": "HD 1234 b",
        "orbital_period": 3.5,
    }

    planet = Planet.from_params(params, required_keys=["name", "orbital_period"])

    with pytest.raises(Exception):
        planet.name = "NewName"

# Tests: Planet.from_params
# Behavior: required_keys empty → no validation → Planet created successfully
def test_planet_required_keys_empty():
    params = {
        "name": "HD 1234 b",
        "orbital_period": 3.5,
    }

    planet = Planet.from_params(params, required_keys=[])

    assert planet.name == "HD 1234 b"
    assert planet.orbital_period == 3.5


# Tests: Planet.from_params
# Behavior: required key present but is_missing() returns True → raises ValueError
def test_planet_required_key_present_but_is_missing(monkeypatch):
    # Patch the reference actually used inside Planet.from_params
    monkeypatch.setattr("domain.planet.is_missing", lambda v: True)

    params = {
        "name": "HD 1234 b",
        "orbital_period": 3.5,
    }

    with pytest.raises(ValueError) as exc:
        Planet.from_params(params, required_keys=["name", "orbital_period"])

    msg = str(exc.value)
    assert "orbital_period" in msg
    assert "missing" in msg.lower()
