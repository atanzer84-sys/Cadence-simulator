"""Tests for loaders.load_stellar_and_planetary_properties."""

from pathlib import Path

import pytest

import configs.global_config as gc
from loaders.load_stellar_and_planetary_properties import (
    apply_log_r_fallback,
    infer_mamajek_spectral_type,
    load_stellar_and_planetary_properties,
    merge_gaia_into_star_params,
)
from loaders import load_stellar_and_planetary_properties as lspp
from tests.helpers.global_config_factory import make_global_cfg


def _dummy_global_cfg(**overrides):
    """Build a minimal GlobalConfig for tests, with sensible defaults and optional overrides."""
    return make_global_cfg(**overrides)


def test_find_excel_file_no_excel(tmp_path: Path) -> None:
    """No *.xlsx files -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        lspp._find_excel_file(tmp_path)

    assert "No Excel file found" in str(exc_info.value)


def test_find_excel_file_single_excel(tmp_path: Path) -> None:
    """Exactly one *.xlsx file -> that path is returned."""
    excel = tmp_path / "Targets_V10p1.xlsx"
    excel.write_bytes(b"")

    found = lspp._find_excel_file(tmp_path)

    assert found == excel


def test_find_excel_file_multiple_excels(tmp_path: Path) -> None:
    """More than one *.xlsx file -> ValueError with names listed."""
    (tmp_path / "A.xlsx").write_bytes(b"")
    (tmp_path / "B.xlsx").write_bytes(b"")

    with pytest.raises(ValueError) as exc_info:
        lspp._find_excel_file(tmp_path)

    msg = str(exc_info.value)
    assert "Multiple Excel files found" in msg
    assert "A.xlsx" in msg
    assert "B.xlsx" in msg


def test_find_excel_file_ignores_excel_lock_files(tmp_path: Path) -> None:
    (tmp_path / "~$Targets_V10p1.xlsx").write_bytes(b"")
    real = tmp_path / "Targets_V10p1.xlsx"
    real.write_bytes(b"")

    found = lspp._find_excel_file(tmp_path)

    assert found == real


def test_load_stellar_and_planetary_properties_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(lspp, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(
        lspp,
        "_find_excel_file",
        lambda repo_root: (_ for _ in ()).throw(FileNotFoundError("no excel")),
    )

    # load_stellar_and_planetary_properties now requires a loaded GlobalConfig.
    dummy_cfg = _dummy_global_cfg()
    monkeypatch.setattr(lspp, "get_global_config", lambda: dummy_cfg)

    with pytest.raises(FileNotFoundError):
        load_stellar_and_planetary_properties("Target")


def test_load_stellar_and_planetary_properties_raises_value_error(monkeypatch):
    monkeypatch.setattr(lspp, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(lspp, "_find_excel_file", lambda repo_root: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(
        lspp,
        "load_matching_excel_row_from_excel",
        lambda _p, _t: (_ for _ in ()).throw(ValueError("bad excel")),
    )

    dummy_cfg = _dummy_global_cfg()
    monkeypatch.setattr(lspp, "get_global_config", lambda: dummy_cfg)

    with pytest.raises(ValueError):
        load_stellar_and_planetary_properties("Target")


def test_infer_mamajek_spectral_type_sets_spectral_type(monkeypatch, caplog):
    class FakeTable:
        def __len__(self):
            return 2

        def __getitem__(self, key):
            if key == "col1":
                return ["F1V", "F2V"]
            if key == "col2":
                return [6900.0, 6800.0]
            raise KeyError(key)

    monkeypatch.setattr(lspp.ascii, "read", lambda *_args, **_kw: FakeTable())

    star_params = {"effective_temperature": 6801.0}
    with caplog.at_level("INFO"):
        out = infer_mamajek_spectral_type(star_params, "dummy_path.txt")

    assert out["spectral_type"] == "F2V"
    assert "Loading Mamajek table" in " ".join(caplog.messages)


def test_infer_mamajek_spectral_type_missing_teff_raises(monkeypatch):
    class FakeTable:
        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key == "col1":
                return ["F1V"]
            if key == "col2":
                return [6900.0]
            raise KeyError(key)

    monkeypatch.setattr(lspp.ascii, "read", lambda *_args, **_kw: FakeTable())

    with pytest.raises(ValueError) as exc_info:
        infer_mamajek_spectral_type({}, "dummy_path.txt")

    assert "effective_temperature" in str(exc_info.value)


def test_infer_mamajek_spectral_type_invalid_teff_raises(monkeypatch):
    class FakeTable:
        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key == "col1":
                return ["F1V"]
            if key == "col2":
                return [6900.0]
            raise KeyError(key)

    monkeypatch.setattr(lspp.ascii, "read", lambda *_args, **_kw: FakeTable())

    with pytest.raises(ValueError) as exc_info:
        infer_mamajek_spectral_type({"effective_temperature": "nope"}, "dummy_path.txt")

    assert "invalid effective_temperature" in str(exc_info.value)


def test_merge_gaia_into_star_params_excel_wins_and_fills_blanks(caplog):
    star_params = {
        "teff": 5000,
        "log_g": "",
        "radius": None,
    }
    gaia_star_params = {
        "teff": 9999,
        "log_g": 4.5,
        "radius": 0.9,
    }

    out = merge_gaia_into_star_params(star_params, gaia_star_params)

    assert out is star_params
    assert star_params["teff"] == 5000
    assert star_params["log_g"] == 4.5
    assert star_params["radius"] == 0.9


def test_merge_gaia_into_star_params_none_gaia_returns_original():
    star_params = {"teff": 5000}

    out = merge_gaia_into_star_params(star_params, None)

    assert out is star_params
    assert star_params == {"teff": 5000}


# --- apply_log_r_fallback fixtures and tests ---


@pytest.fixture
def global_cfg_log_r(monkeypatch):
    cfg = _dummy_global_cfg(
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
        write_intermediate_arrays=True,
        produce_flux_convolution_plots=False,
        cosmic_rays_min=5,
        cosmic_rays_max=10,
        cosmic_ray_signal_electrons=72000,
        cosmic_ray_length_min_px=10,
        cosmic_ray_length_max_px=20,
    )
    monkeypatch.setattr(gc, "_GLOBAL", cfg, raising=False)
    return cfg


@pytest.fixture
def global_cfg_log_r_disabled(monkeypatch):
    cfg = _dummy_global_cfg(
        enable_log_r_fallback=False,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
        write_intermediate_arrays=True,
        produce_flux_convolution_plots=False,
        cosmic_rays_min=5,
        cosmic_rays_max=10,
        cosmic_ray_signal_electrons=72000,
        cosmic_ray_length_min_px=10,
        cosmic_ray_length_max_px=20,
    )
    monkeypatch.setattr(gc, "_GLOBAL", cfg, raising=False)
    return cfg


def test_apply_log_r_fallback_disabled_does_not_modify_dict(global_cfg_log_r_disabled):
    star_params = {"effective_temperature": 6000}
    out = apply_log_r_fallback(star_params, cfg=global_cfg_log_r_disabled)

    assert out is star_params
    assert "log_r" not in star_params


def test_apply_log_r_fallback_does_not_override_existing_log_r(global_cfg_log_r):
    star_params = {"effective_temperature": 6000, "log_r": -9.9}

    out = apply_log_r_fallback(star_params, cfg=global_cfg_log_r)

    assert out is star_params
    assert star_params["log_r"] == -9.9


def test_apply_log_r_fallback_sets_hot_or_cool_value_based_on_threshold(global_cfg_log_r):
    star_hot = {"effective_temperature": "6000"}
    out_hot = apply_log_r_fallback(star_hot, cfg=global_cfg_log_r)
    assert out_hot is star_hot
    assert star_hot["log_r"] == -4.2

    star_cool = {"effective_temperature": "5000"}
    out_cool = apply_log_r_fallback(star_cool, cfg=global_cfg_log_r)
    assert out_cool is star_cool
    assert star_cool["log_r"] == -4.8


def test_apply_log_r_fallback_missing_teff_does_not_modify_dict(global_cfg_log_r):
    star_params = {}
    out = apply_log_r_fallback(star_params, cfg=global_cfg_log_r)

    assert out is star_params
    assert "log_r" not in star_params


def test_apply_log_r_fallback_invalid_teff_does_not_modify_dict(global_cfg_log_r):
    star_params = {"effective_temperature": "nope"}
    out = apply_log_r_fallback(star_params, cfg=global_cfg_log_r)

    assert out is star_params
    assert "log_r" not in star_params
