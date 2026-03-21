import pytest

from loaders.parameter_preprocessing import is_missing, get_missing_properties, clean_and_cast_parameters
from domain.star import Star
from domain.planet import Planet


class TestIsMissing:
    # Tests: is_missing
    # Behavior: treats None and empty/whitespace strings as missing
    @pytest.mark.parametrize("value", [None, "", "   ", "\t", "\n"])
    def test_is_missing_true_for_none_or_whitespace_strings(self, value):
        assert is_missing(value) is True

    # Tests: is_missing
    # Behavior: keeps non-string falsy values as not missing
    @pytest.mark.parametrize("value", [0, 0.0, False, [], {}])
    def test_is_missing_false_for_non_string_falsy_values(self, value):
        assert is_missing(value) is False


class TestGetMissingProperties:
    # Tests: get_missing_properties
    # Behavior: returns empty list when all required keys are present
    def test_returns_empty_list_when_all_required_present(self):
        parameters = {"a": 1, "b": "x", "c": 0.0}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == []

    # Tests: get_missing_properties
    # Behavior: marks missing keys that are absent in parameters
    def test_missing_when_key_absent(self):
        parameters = {"a": 1}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["b"]

    # Tests: get_missing_properties
    # Behavior: marks keys with None values as missing
    def test_missing_when_value_is_none(self):
        parameters = {"a": None, "b": "x"}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["a"]

    # Tests: get_missing_properties
    # Behavior: marks empty and whitespace-only strings as missing
    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_missing_when_value_is_empty_or_whitespace_string(self, value):
        parameters = {"a": value, "b": "ok"}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["a"]

    # Tests: get_missing_properties
    # Behavior: preserves required_keys order in output
    def test_order_matches_required_keys_order(self):
        parameters = {"a": None, "c": None}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == ["a", "b", "c"]

    # Tests: get_missing_properties
    # Behavior: non-string zero-like values are not treated as missing
    def test_non_string_zero_is_not_missing(self):
        parameters = {"a": 0, "b": 0.0, "c": False}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == []

    # Tests: get_missing_properties
    # Behavior: log_output=False suppresses info log messages
    def test_log_output_false_suppresses_logging(self, caplog):
        parameters = {"a": None}
        required = ["a"]
        with caplog.at_level("INFO"):
            out = get_missing_properties(parameters, required, log_output=False)
        assert out == ["a"]
        assert not caplog.records


class TestCleanAndCastParameters:
    # Tests: clean_and_cast_parameters
    # Behavior: does not mutate input dictionary
    def test_does_not_mutate_input_dict(self):
        original = {"name": "WASP-189", "effective_temperature": "5778"}
        before = dict(original)
        _ = clean_and_cast_parameters(original, Star)
        assert original == before

    # Tests: clean_and_cast_parameters
    # Behavior: strips string values and converts empty strings to None
    def test_strips_strings_and_converts_empty_to_none(self):
        params = {
            "name": "  WASP-189  ",
            "spectral_type": "  A6IV-V  ",
            "some_unknown_str": "   ",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["name"] == "WASP-189"
        assert cleaned["spectral_type"] == "A6IV-V"
        assert cleaned["some_unknown_str"] is None

    # Tests: clean_and_cast_parameters
    # Behavior: casts float-typed dataclass fields from strings
    def test_casts_float_fields_from_string(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": "  8000  ",
            "radius": "2.36",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0
        assert cleaned["radius"] == 2.36

    # Tests: clean_and_cast_parameters
    # Behavior: casts float-typed dataclass fields from ints
    def test_casts_float_fields_from_int(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": 8000,
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0

    # Tests: clean_and_cast_parameters
    # Behavior: keeps None values for dataclass fields unchanged
    def test_keeps_none_as_none_for_model_fields(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": None,
            "radius": "   ",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] is None
        assert cleaned["radius"] is None

    # Tests: clean_and_cast_parameters
    # Behavior: casts str-typed dataclass fields using str(...)
    def test_casts_str_fields_to_str(self):
        params = {
            "name": 12345,  # should become "12345"
            "spectral_type": 7,  # should become "7"
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["name"] == "12345"
        assert cleaned["spectral_type"] == "7"

    # Tests: clean_and_cast_parameters
    # Behavior: leaves unknown keys untouched except generic string normalization
    def test_unknown_keys_are_left_untouched_except_string_normalization(self):
        params = {
            "name": "WASP-189",
            "unknown_numeric_like": "12.3",  # unknown key -> should remain "12.3" (string)
            "unknown_int": 99,               # unknown key -> stays int
            "unknown_none": None,            # unknown key -> stays None
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["unknown_numeric_like"] == "12.3"
        assert cleaned["unknown_int"] == 99
        assert cleaned["unknown_none"] is None

    # Tests: clean_and_cast_parameters
    # Behavior: raises ValueError with useful message and logs on invalid float cast
    def test_invalid_float_raises_valueerror_with_key_and_value_and_logs(self, caplog):
        params = {
            "name": "WASP-189",
            "effective_temperature": "mus",
        }
        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError) as excinfo:
                _ = clean_and_cast_parameters(params, Star)

        msg = str(excinfo.value)
        assert "effective_temperature" in msg
        assert "mus" in msg

        # Ensure we logged something meaningful
        assert any(
            "Invalid numeric value for parameter 'effective_temperature'" in record.message
            for record in caplog.records
        )

    # Tests: clean_and_cast_parameters
    # Behavior: applies same normalization and casting rules for Planet dataclass
    def test_planet_casting_works_the_same_way(self):
        params = {
            "name": "WASP-189 b",
            "discoverymethod": "  Transit  ",
            "orbital_period": "  2.724033  ",
            "orbital_semi_major_axis": "0.05053",
            "radius_jupiter": "1.619",
            "mass_jupiter": "1.99",
            "equilibrium_temperature": "3353",
            "scale_height": "1130.2166",
        }
        cleaned = clean_and_cast_parameters(params, Planet)
        assert cleaned["discoverymethod"] == "Transit"
        assert cleaned["orbital_period"] == 2.724033
        assert cleaned["orbital_semi_major_axis"] == 0.05053
        assert cleaned["radius_jupiter"] == 1.619
        assert cleaned["mass_jupiter"] == 1.99
        assert cleaned["equilibrium_temperature"] == 3353.0
        assert cleaned["scale_height"] == pytest.approx(1130.2166)

    # Tests: clean_and_cast_parameters
    # Behavior: handles extra keys outside dataclass without crashing
    def test_handles_parameters_not_in_dataclass_without_crashing(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": "8000",
            "totally_new_key": "  keep me  ",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0
        assert cleaned["totally_new_key"] == "keep me"


# Tests: clean_and_cast_parameters
# Behavior: supports typing.Union and PEP604 optional float annotations
def test_clean_and_cast_parameters_handles_union_and_pep604_float_fields(monkeypatch):
    from dataclasses import dataclass
    from typing import Union

    @dataclass
    class Dummy:
        a: Union[float, None]      # typing.Union
        b: float | None            # PEP604

    params = {"a": "1.5", "b": "2.5"}
    cleaned = clean_and_cast_parameters(params, Dummy)

    assert cleaned["a"] == 1.5
    assert cleaned["b"] == 2.5
