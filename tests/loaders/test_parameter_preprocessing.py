import pytest

from loaders.parameter_preprocessing import get_missing_properties, clean_and_cast_parameters
from domain.star import Star
from domain.planet import Planet


class TestGetMissingProperties:
    def test_returns_empty_list_when_all_required_present(self):
        parameters = {"a": 1, "b": "x", "c": 0.0}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == []

    def test_missing_when_key_absent(self):
        parameters = {"a": 1}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["b"]

    def test_missing_when_value_is_none(self):
        parameters = {"a": None, "b": "x"}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["a"]

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_missing_when_value_is_empty_or_whitespace_string(self, value):
        parameters = {"a": value, "b": "ok"}
        required = ["a", "b"]
        assert get_missing_properties(parameters, required) == ["a"]

    def test_order_matches_required_keys_order(self):
        parameters = {"a": None, "c": None}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == ["a", "b", "c"]

    def test_non_string_zero_is_not_missing(self):
        parameters = {"a": 0, "b": 0.0, "c": False}
        required = ["a", "b", "c"]
        assert get_missing_properties(parameters, required) == []


class TestCleanAndCastParameters:
    def test_does_not_mutate_input_dict(self):
        original = {"name": "WASP-189", "effective_temperature": "5778"}
        before = dict(original)
        _ = clean_and_cast_parameters(original, Star)
        assert original == before

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

    def test_casts_float_fields_from_string(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": "  8000  ",
            "radius": "2.36",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0
        assert cleaned["radius"] == 2.36

    def test_casts_float_fields_from_int(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": 8000,
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0

    def test_keeps_none_as_none_for_model_fields(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": None,
            "radius": "   ",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] is None
        assert cleaned["radius"] is None

    def test_casts_str_fields_to_str(self):
        params = {
            "name": 12345,  # should become "12345"
            "spectral_type": 7,  # should become "7"
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["name"] == "12345"
        assert cleaned["spectral_type"] == "7"

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

    def test_handles_parameters_not_in_dataclass_without_crashing(self):
        params = {
            "name": "WASP-189",
            "effective_temperature": "8000",
            "totally_new_key": "  keep me  ",
        }
        cleaned = clean_and_cast_parameters(params, Star)
        assert cleaned["effective_temperature"] == 8000.0
        assert cleaned["totally_new_key"] == "keep me"
