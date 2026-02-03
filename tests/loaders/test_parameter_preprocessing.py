"""Tests for parameter_preprocessing: process_*_parameter_values."""

import pytest

from loaders.parameter_preprocessing import (
    process_planetary_parameter_values,
    process_stellar_parameter_values,
)


def test_process_stellar_parameters_missing_name_raises() -> None:
    params = {"st_teff": "5000"}
    with pytest.raises(ValueError, match="st_name"):
        process_stellar_parameter_values(params)


def test_process_stellar_parameters_name_none_raises() -> None:
    params = {"st_name": None, "st_teff": "5000"}
    with pytest.raises(ValueError, match="st_name"):
        process_stellar_parameter_values(params)


def test_process_planetary_parameters_missing_name_raises() -> None:
    params = {"pl_orbper": "3.4"}
    with pytest.raises(ValueError, match="pl_name"):
        process_planetary_parameter_values(params)


def test_process_planetary_parameters_name_none_raises() -> None:
    params = {"pl_name": None, "pl_orbper": "3.4"}
    with pytest.raises(ValueError, match="pl_name"):
        process_planetary_parameter_values(params)


@pytest.mark.parametrize("key", [
    "st_teff",
    "st_rad",
    "st_mass",
    "st_logg",
    "st_met",
    "st_dist",
    "st_vsin",
    "st_lum",
])
def test_process_stellar_parameters_empty_string_becomes_none(key: str) -> None:
    params = {"st_name": "Star", key: "   "}
    result = process_stellar_parameter_values(params)
    assert result[key] is None


@pytest.mark.parametrize("key,value", [
    ("st_teff", "5000"),
    ("st_rad", "1.1"),
    ("st_mass", "0.9"),
    ("st_logg", "4.3"),
    ("st_met", "-0.1"),
    ("st_dist", "123.4"),
    ("st_vsin", "5.2"),
    ("st_lum", "2.5"),
])
def test_process_stellar_parameters_numeric_strings_converted(
    key: str,
    value: str,
) -> None:
    params = {"st_name": "Star", key: value}
    result = process_stellar_parameter_values(params)
    assert result[key] == float(value)


@pytest.mark.parametrize("key", [
    "st_teff",
    "st_rad",
    "st_mass",
    "st_logg",
    "st_met",
    "st_dist",
    "st_vsin",
    "st_lum",
])
def test_process_stellar_parameters_invalid_numeric_raises(key: str) -> None:
    params = {"st_name": "Star", key: "not_a_number"}
    with pytest.raises(ValueError, match=key):
        process_stellar_parameter_values(params)


@pytest.mark.parametrize("key", [
    "pl_orbper",
    "pl_orbsmax",
    "pl_radj",
    "pl_bmassj",
    "pl_eqt",
    "scale_height_km",
])
def test_process_planetary_parameters_empty_string_becomes_none(key: str) -> None:
    params = {"pl_name": "Planet", key: "   "}
    result = process_planetary_parameter_values(params)
    assert result[key] is None


@pytest.mark.parametrize("key,value", [
    ("pl_orbper", "3.4"),
    ("pl_orbsmax", "0.05"),
    ("pl_radj", "1.2"),
    ("pl_bmassj", "0.7"),
    ("pl_eqt", "1000"),
    ("scale_height_km", "250"),
])
def test_process_planetary_parameters_numeric_strings_converted(
    key: str,
    value: str,
) -> None:
    params = {"pl_name": "Planet", key: value}
    result = process_planetary_parameter_values(params)
    assert result[key] == float(value)


@pytest.mark.parametrize("key", [
    "pl_orbper",
    "pl_orbsmax",
    "pl_radj",
    "pl_bmassj",
    "pl_eqt",
    "scale_height_km",
])
def test_process_planetary_parameters_invalid_numeric_raises(key: str) -> None:
    params = {"pl_name": "Planet", key: "not_a_number"}
    with pytest.raises(ValueError, match=key):
        process_planetary_parameter_values(params)
