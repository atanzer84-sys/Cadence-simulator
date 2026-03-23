"""Tests for loaders.load_stellar_and_planetary_properties."""

from pathlib import Path
import sys

import pytest

from loaders.load_stellar_and_planetary_properties import (
    _find_excel_file,
    apply_log_r,
    infer_mamajek_spectral_type,
    infer_mamajek,
    load_excel_mapping,
    load_stellar_and_planetary_properties,
    merge_gaia_into_star_params,
    apply_radius_from_teff_mag_distance_if_missing,
)


def _make_mamajek_table(spectral_types, temperatures):
    class FakeTable:
        def __len__(self):
            return len(spectral_types)

        def __getitem__(self, key):
            if key == "col1":
                return spectral_types
            if key == "col2":
                return temperatures
            raise KeyError(key)

    return FakeTable()


# Tests: _find_excel_file
# Behavior: raises when no Excel file exists
def test_find_excel_file_no_excel(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _find_excel_file(tmp_path)


# Tests: _find_excel_file
# Behavior: returns the only Excel file
def test_find_excel_file_single_excel(tmp_path: Path):
    excel = tmp_path / "Targets_V10p1.xlsx"
    excel.write_bytes(b"")

    found = _find_excel_file(tmp_path)

    assert found == excel


# Tests: _find_excel_file
# Behavior: raises when multiple Excel files exist
def test_find_excel_file_multiple_excels(tmp_path: Path):
    (tmp_path / "A.xlsx").write_bytes(b"")
    (tmp_path / "B.xlsx").write_bytes(b"")

    with pytest.raises(ValueError):
        _find_excel_file(tmp_path)


# Tests: _find_excel_file
# Behavior: ignores temporary Excel lock files
def test_find_excel_file_ignores_excel_lock_files(tmp_path: Path):
    (tmp_path / "~$Targets_V10p1.xlsx").write_bytes(b"")
    real = tmp_path / "Targets_V10p1.xlsx"
    real.write_bytes(b"")

    found = _find_excel_file(tmp_path)

    assert found == real


# Tests: load_excel_mapping
# Behavior: resolves repo-root mapping path and forwards to load_excel_cfg
def test_load_excel_mapping_uses_repo_root_configs_path(monkeypatch):
    mod = sys.modules[load_stellar_and_planetary_properties.__module__]
    recorded = {"path": None}

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(mod, "load_excel_cfg", lambda p: recorded.__setitem__("path", p) or {"ok": True})

    out = load_excel_mapping()
    assert out == {"ok": True}
    assert recorded["path"] == Path("/repo") / "configs" / "excel_mapping.cfg"


# Tests: infer_mamajek
# Behavior: resolves Mamajek table path and delegates to infer_mamajek_spectral_type
def test_infer_mamajek_delegates_with_repo_path(monkeypatch):
    mod = sys.modules[infer_mamajek.__module__]
    recorded = {"star_params": None, "path": None, "log_output": None}

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/repo"))

    def _fake_infer(star_params, mamajek_path, log_output=True):
        recorded["star_params"] = star_params
        recorded["path"] = mamajek_path
        recorded["log_output"] = log_output
        return {"spectral_type": "G2V"}

    monkeypatch.setattr(mod, "infer_mamajek_spectral_type", _fake_infer)

    out = infer_mamajek({"effective_temperature": 5777.0}, log_output=False)

    assert out == {"spectral_type": "G2V"}
    assert recorded["star_params"] == {"effective_temperature": 5777.0}
    assert recorded["path"] == Path("/repo") / "data" / "stellar_param_mamjeck.txt"
    assert recorded["log_output"] is False


# Tests: load_stellar_and_planetary_properties
# Behavior: raises when the Excel file cannot be found
def test_load_stellar_and_planetary_properties_raises_file_not_found(monkeypatch, make_global_config):
    mod = sys.modules[load_stellar_and_planetary_properties.__module__]
    cfg = make_global_config()

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/tmp"))
    monkeypatch.setattr(mod, "_find_excel_file", lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(mod, "get_global_config", lambda: cfg)

    with pytest.raises(FileNotFoundError):
        load_stellar_and_planetary_properties("Target")


# Tests: load_stellar_and_planetary_properties
# Behavior: raises when row loading fails
def test_load_stellar_and_planetary_properties_raises_value_error(monkeypatch, make_global_config):
    mod = sys.modules[load_stellar_and_planetary_properties.__module__]
    cfg = make_global_config()

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/tmp"))
    monkeypatch.setattr(mod, "_find_excel_file", lambda _: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(mod, "load_matching_excel_row_from_excel", lambda *_: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(mod, "get_global_config", lambda: cfg)

    with pytest.raises(ValueError):
        load_stellar_and_planetary_properties("Target")


# Tests: infer_mamajek_spectral_type
# Behavior: sets the nearest spectral type
def test_infer_mamajek_spectral_type_sets_spectral_type(monkeypatch):
    mod = sys.modules[infer_mamajek_spectral_type.__module__]
    table = _make_mamajek_table(["F1V", "F2V"], [6900.0, 6800.0])

    monkeypatch.setattr(mod.ascii, "read", lambda *args, **kwargs: table)
    monkeypatch.setattr(mod, "_MAMAJEK_CACHE", None)

    star_params = {"effective_temperature": 6801.0}
    out = infer_mamajek_spectral_type(star_params, "dummy", log_output=False)

    assert out["spectral_type"] == "F2V"


# Tests: infer_mamajek_spectral_type
# Behavior: raises when effective temperature is missing
def test_infer_mamajek_spectral_type_missing_teff_raises(monkeypatch):
    mod = sys.modules[infer_mamajek_spectral_type.__module__]
    table = _make_mamajek_table(["F1V"], [6900.0])

    monkeypatch.setattr(mod.ascii, "read", lambda *args, **kwargs: table)
    monkeypatch.setattr(mod, "_MAMAJEK_CACHE", None)

    with pytest.raises(ValueError):
        infer_mamajek_spectral_type({}, "dummy")


# Tests: infer_mamajek_spectral_type
# Behavior: raises for invalid effective temperature
def test_infer_mamajek_spectral_type_invalid_teff_raises(monkeypatch):
    mod = sys.modules[infer_mamajek_spectral_type.__module__]
    table = _make_mamajek_table(["F1V"], [6900.0])

    monkeypatch.setattr(mod.ascii, "read", lambda *args, **kwargs: table)
    monkeypatch.setattr(mod, "_MAMAJEK_CACHE", None)

    with pytest.raises(ValueError):
        infer_mamajek_spectral_type({"effective_temperature": "nope"}, "dummy")


# Tests: infer_mamajek_spectral_type
# Behavior: reloads Mamajek table when file path changes
def test_infer_mamajek_spectral_type_reloads_when_path_changes(monkeypatch):
    mod = sys.modules[infer_mamajek_spectral_type.__module__]
    reads = {"count": 0}

    def fake_read(path, comment="#"):
        reads["count"] += 1
        if str(path).endswith("a.txt"):
            return _make_mamajek_table(["F1V"], [6900.0])
        return _make_mamajek_table(["K5V"], [4400.0])

    monkeypatch.setattr(mod.ascii, "read", fake_read)
    monkeypatch.setattr(mod, "_MAMAJEK_CACHE", None)

    out_a = infer_mamajek_spectral_type({"effective_temperature": 6900.0}, "a.txt", log_output=False)
    out_b = infer_mamajek_spectral_type({"effective_temperature": 4400.0}, "b.txt", log_output=False)

    assert reads["count"] == 2
    assert out_a["spectral_type"] == "F1V"
    assert out_b["spectral_type"] == "K5V"


# Tests: merge_gaia_into_star_params
# Behavior: keeps Excel values and fills blanks
def test_merge_gaia_into_star_params_excel_wins_and_fills_blanks():
    star_params = {"teff": 5000, "log_g": "", "radius": None}
    gaia_params = {"teff": 9999, "log_g": 4.5, "radius": 0.9}

    out = merge_gaia_into_star_params(star_params, gaia_params)

    assert out is star_params
    assert star_params == {"teff": 5000, "log_g": 4.5, "radius": 0.9}


# Tests: merge_gaia_into_star_params
# Behavior: returns the original dict when Gaia data is missing
def test_merge_gaia_into_star_params_none_gaia_returns_original():
    star_params = {"teff": 5000}

    out = merge_gaia_into_star_params(star_params, None)

    assert out is star_params
    assert star_params == {"teff": 5000}


# Tests: apply_log_r
# Behavior: leaves the dict unchanged when fallback is disabled
def test_apply_log_r_disabled(make_global_config):
    cfg = make_global_config(enable_log_r_fallback=False)
    star_params = {"effective_temperature": 6000}

    out = apply_log_r(star_params, cfg)

    assert out is star_params
    assert star_params == {"effective_temperature": 6000}


# Tests: apply_log_r
# Behavior: keeps an existing log_r value
def test_apply_log_r_does_not_override_existing_log_r(make_global_config):
    cfg = make_global_config(
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
    )
    star_params = {"effective_temperature": 6000, "log_r": -9.9}

    out = apply_log_r(star_params, cfg)

    assert out is star_params
    assert star_params == {"effective_temperature": 6000, "log_r": -9.9}


# Tests: apply_log_r
# Behavior: sets hot and cool fallback values
def test_apply_log_r_sets_hot_or_cool_value_based_on_threshold(make_global_config):
    cfg = make_global_config(
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
    )

    star_hot = {"effective_temperature": "6000"}
    out_hot = apply_log_r(star_hot, cfg)

    assert out_hot is star_hot
    assert star_hot == {"effective_temperature": "6000", "log_r": -4.2}

    star_cool = {"effective_temperature": "5000"}
    out_cool = apply_log_r(star_cool, cfg)

    assert out_cool is star_cool
    assert star_cool == {"effective_temperature": "5000", "log_r": -4.8}


# Tests: apply_log_r
# Behavior: leaves the dict unchanged when temperature is missing
def test_apply_log_r_missing_teff_does_not_modify_dict(make_global_config):
    cfg = make_global_config(
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
    )
    star_params = {}

    out = apply_log_r(star_params, cfg)

    assert out is star_params
    assert star_params == {}


# Tests: apply_log_r
# Behavior: leaves the dict unchanged for invalid temperature
def test_apply_log_r_invalid_teff_does_not_modify_dict(make_global_config):
    cfg = make_global_config(
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
    )
    star_params = {"effective_temperature": "nope"}

    out = apply_log_r(star_params, cfg)

    assert out is star_params
    assert star_params == {"effective_temperature": "nope"}

# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: keeps existing radius unchanged
def test_apply_radius_from_teff_mag_distance_if_missing_keeps_existing_radius():
    star_params = {
        "effective_temperature": 5777.0,
        "gaia_magnitude": 10.0,
        "distance": 100.0,
        "radius": 1.23,
    }

    out = apply_radius_from_teff_mag_distance_if_missing(star_params)

    assert out is star_params
    assert star_params == {
        "effective_temperature": 5777.0,
        "gaia_magnitude": 10.0,
        "distance": 100.0,
        "radius": 1.23,
    }


# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: computes radius from valid temperature magnitude and distance
def test_apply_radius_from_teff_mag_distance_if_missing_computes_radius():
    star_params = {
        "effective_temperature": 5777.0,
        "gaia_magnitude": 4.83,
        "distance": 10.0,
    }

    out = apply_radius_from_teff_mag_distance_if_missing(star_params)

    assert out is star_params
    assert "radius" in star_params
    assert star_params["radius"] is not None
    assert star_params["radius"] > 0.0


# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: returns the same dictionary object (contract)
def test_apply_radius_from_teff_mag_distance_if_missing_returns_dict_contract():
    star_params = {
        "effective_temperature": 5777.0,
        "gaia_magnitude": 4.83,
        "distance": 10.0,
    }
    out = apply_radius_from_teff_mag_distance_if_missing(star_params)
    assert out is star_params


# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: leaves values unchanged when required inputs are missing
@pytest.mark.parametrize(
    "star_params",
    [
        {"gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0},
        {"effective_temperature": None, "gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": None, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0, "distance": None},
    ],
)
def test_apply_radius_from_teff_mag_distance_if_missing_leaves_missing_inputs_unchanged(star_params):
    original = dict(star_params)

    out = apply_radius_from_teff_mag_distance_if_missing(star_params)

    assert out is star_params
    assert star_params == original


# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: rejects invalid numeric input without mutating values
@pytest.mark.parametrize(
    "star_params",
    [
        {"effective_temperature": "nope", "gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": "garbage", "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0, "distance": "banana"},
        {"effective_temperature": "", "gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": "   ", "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0, "distance": object()},
        {"effective_temperature": -100.0, "gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0, "distance": -5.0},
        {"effective_temperature": 0.0, "gaia_magnitude": 10.0, "distance": 100.0},
        {"effective_temperature": 5777.0, "gaia_magnitude": 10.0, "distance": 0.0},
    ],
)
def test_apply_radius_from_teff_mag_distance_if_missing_rejects_bullshit_input(star_params):
    original = dict(star_params)

    out = apply_radius_from_teff_mag_distance_if_missing(star_params)

    assert out is star_params
    assert star_params == original


# Tests: apply_radius_from_teff_mag_distance_if_missing
# Behavior: returns unchanged params when computed radius is non-finite
def test_apply_radius_from_teff_mag_distance_if_missing_non_finite_radius_returns_unchanged():
    star_params = {
        "effective_temperature": 5777.0,
        "gaia_magnitude": float("nan"),
        "distance": 10.0,
    }
    original = dict(star_params)

    out = apply_radius_from_teff_mag_distance_if_missing(star_params)

    assert out is star_params
    assert star_params == original
    assert "radius" not in star_params


# Tests: merge_gaia_into_star_params
# Behavior: fills whitespace only Excel values from Gaia
def test_merge_gaia_into_star_params_fills_whitespace_only_values():
    star_params = {
        "teff": "   ",
        "radius": "",
        "distance": None,
    }
    gaia_params = {
        "teff": 5777.0,
        "radius": 1.0,
        "distance": 10.0,
    }

    out = merge_gaia_into_star_params(star_params, gaia_params)

    assert out is star_params
    assert star_params == {
        "teff": 5777.0,
        "radius": 1.0,
        "distance": 10.0,
    }


# Tests: merge_gaia_into_star_params
# Behavior: fills blank Excel values with Gaia values, including None
def test_merge_gaia_into_star_params_fills_blank_values_with_none():
    star_params = {
        "teff": "",
        "radius": None,
    }
    gaia_params = {
        "teff": None,
        "radius": None,
    }

    out = merge_gaia_into_star_params(star_params, gaia_params)

    assert out is star_params
    assert star_params == {
        "teff": None,
        "radius": None,
    }

# Tests: merge_gaia_into_star_params
# Behavior: adds Gaia keys that are not present in the Excel dict
def test_merge_gaia_into_star_params_adds_unknown_gaia_keys():
    star_params = {"teff": 5000}
    gaia_params = {"radius": 1.0, "metallicity": 0.2}

    out = merge_gaia_into_star_params(star_params, gaia_params)

    assert out is star_params
    assert star_params == {
        "teff": 5000,
        "radius": 1.0,
        "metallicity": 0.2,
    }


# Tests: load_stellar_and_planetary_properties
# Behavior: raises when required stellar values remain missing after all fallbacks
def test_load_stellar_and_planetary_properties_raises_when_required_star_values_remain_missing(monkeypatch, make_global_config):
    mod = sys.modules[load_stellar_and_planetary_properties.__module__]
    cfg = make_global_config()

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/tmp"))
    monkeypatch.setattr(mod, "_find_excel_file", lambda _: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(mod, "load_matching_excel_row_from_excel", lambda *_: ({"pl_name": "Target b"}, "Target"))
    monkeypatch.setattr(
        mod,
        "map_to_planet_or_star_dictionary",
        lambda planet_star_dictionary, mapping, target_name: (
            {"name": "Target b"},
            {"name": "Target", "right_ascension": 1.0, "declination": 2.0},
        ),
    )
    monkeypatch.setattr(mod, "get_global_config", lambda: cfg)
    monkeypatch.setattr(mod, "lookup_target_star_gaia", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "merge_gaia_into_star_params", lambda star_params, gaia_params: star_params)
    monkeypatch.setattr(mod, "apply_radius_from_teff_mag_distance_if_missing", lambda star_params: star_params)
    monkeypatch.setattr(mod, "infer_mamajek", lambda star_params, log_output=True: star_params)
    monkeypatch.setattr(mod, "apply_log_r", lambda star_params, cfg, log_output=True: star_params)
    monkeypatch.setattr(mod, "load_excel_mapping", lambda: {"required_stellar_parameters": ["name", "radius"]})
    monkeypatch.setattr(mod, "get_missing_properties", lambda star_params, required_keys, log_output=False: ["radius"])

    with pytest.raises(ValueError):
        load_stellar_and_planetary_properties("Target")




# Tests: load_stellar_and_planetary_properties
# Behavior: returns merged planet and star values on the happy path
def test_load_stellar_and_planetary_properties_happy_path(monkeypatch, make_global_config):
    mod = sys.modules[load_stellar_and_planetary_properties.__module__]
    cfg = make_global_config()

    monkeypatch.setattr(mod, "get_repo_root", lambda: Path("/tmp"))
    monkeypatch.setattr(mod, "_find_excel_file", lambda _: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(
        mod,
        "load_matching_excel_row_from_excel",
        lambda *_: ({"pl_name": "Target b"}, "Target"),
    )
    monkeypatch.setattr(
        mod,
        "map_to_planet_or_star_dictionary",
        lambda planet_star_dictionary, mapping, target_name: (
            {"name": "Target b"},
            {"name": "Target", "right_ascension": 1.0, "declination": 2.0, "radius": None},
        ),
    )
    monkeypatch.setattr(mod, "get_global_config", lambda: cfg)
    monkeypatch.setattr(mod, "lookup_target_star_gaia", lambda *args, **kwargs: {"radius": 1.0})
    monkeypatch.setattr(mod, "merge_gaia_into_star_params", lambda star_params, gaia_params: {**star_params, **gaia_params})
    monkeypatch.setattr(mod, "apply_radius_from_teff_mag_distance_if_missing", lambda star_params: star_params)
    monkeypatch.setattr(mod, "infer_mamajek", lambda star_params, log_output=True: star_params)
    monkeypatch.setattr(mod, "apply_log_r", lambda star_params, cfg, log_output=True: star_params)
    monkeypatch.setattr(
        mod,
        "load_excel_mapping",
        lambda: {
            "required_stellar_parameters": ["name", "radius"],
            "required_planetary_parameters": ["name"],
        },
    )

    calls = {"count": 0}

    def fake_get_missing_properties(star_params, required_keys, log_output=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return ["radius"]
        return []

    monkeypatch.setattr(mod, "get_missing_properties", fake_get_missing_properties)
    monkeypatch.setattr(mod, "clean_and_cast_parameters", lambda params, cls: params)

    planet_params, star_params, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties("Target")

    assert planet_params == {"name": "Target b"}
    assert star_params == {
        "name": "Target",
        "right_ascension": 1.0,
        "declination": 2.0,
        "radius": 1.0,
    }
    assert required_planetary_parameters == ["name"]
    assert required_stellar_parameters == ["name", "radius"]