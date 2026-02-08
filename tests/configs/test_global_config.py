from __future__ import annotations

from pathlib import Path
import pytest

import configs.global_config as global_config


@pytest.fixture(autouse=True)
def reset_cache():
    # reset the cache in the module under test
    global_config._GLOBAL = None
    yield
    global_config._GLOBAL = None


def _cfg_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def test_get_global_config_raises_if_not_loaded(caplog):
    with pytest.raises(RuntimeError) as exc:
        global_config.get_global_config()

    assert "Global config not loaded" in str(exc.value)
    assert any(
        "Global config not loaded" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_load_global_config_returns_globalconfig_instance():
    cfg = global_config.load_global_config(_cfg_path("global_minimal.cfg"))
    assert isinstance(cfg, global_config.GlobalConfig)


def test_load_global_config_is_cached_and_shared():
    cfg1 = global_config.load_global_config(_cfg_path("global_minimal.cfg"))
    cfg2 = global_config.load_global_config(_cfg_path("global_minimal.cfg"))

    assert cfg1 is cfg2
    assert global_config.get_global_config() is cfg1


def test_optional_float_fields_accept_blank_and_none():
    cfg = global_config.load_global_config(_cfg_path("global_optional_blanks.cfg"))

    assert cfg.mg2_col is None
    assert cfg.mg1_col is None
    assert cfg.fe2_col is None


def test_boolean_fields_accept_common_spellings():
    cfg = global_config.load_global_config(_cfg_path("global_optional_blanks.cfg"))

    assert cfg.line_core_emission is False
    assert cfg.interstellar_absorption is True
    assert cfg.test_mode is False


def test_missing_config_file_logs_and_raises(caplog, tmp_path):
    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        global_config.load_global_config(missing)

    assert any(
        "not found" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_invalid_boolean_reports_property_name(caplog):
    with pytest.raises(ValueError) as exc:
        global_config.load_global_config(_cfg_path("global_invalid_bool.cfg"))

    msg = str(exc.value)
    assert "line_core_emission" in msg
    assert "Invalid boolean value" in msg

    assert any(
        "line_core_emission" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )

    
def test_missing_sigmaMg22_uses_default_and_logs_warning(caplog, tmp_path):
    cfg_path = tmp_path / "global_missing_sigma_h.cfg"
    cfg_path.write_text(
        """
line_core_emission = 1
interstellar_absorption = 0
sigmaMg21 = 0.288
test_mode = 0
""",
        encoding="utf-8",
    )

    cfg = global_config.load_global_config(cfg_path)

    assert cfg.sigmaMg22 == global_config.DEFAULT_SIGMA_MG22
    assert any("sigmaMg22 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)

def test_optional_float_parses_numeric(tmp_path):
    cfg_path = tmp_path / "global_numeric_optional.cfg"
    cfg_path.write_text(
        """
line_core_emission = 0
interstellar_absorption = 0
mg2_col = 12.5
mg1_col = 3
fe2_col = 0.001
test_mode = 0
produce_Plots = 0
""",
        encoding="utf-8",
    )

    cfg = global_config.load_global_config(cfg_path)

    assert cfg.mg2_col == 12.5
    assert cfg.mg1_col == 3.0
    assert cfg.fe2_col == 0.001

def test_optional_float_invalid_value_raises(tmp_path):
    cfg_path = tmp_path / "global_invalid_optional.cfg"
    cfg_path.write_text(
        """
line_core_emission = 0
interstellar_absorption = 0
mg2_col = not_a_number
test_mode = 0
produce_Plots = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        global_config.load_global_config(cfg_path)

def test_invalid_float_reports_property_name(caplog, tmp_path):
    cfg_path = tmp_path / "global_invalid_float.cfg"
    cfg_path.write_text(
        """
line_core_emission = 0
interstellar_absorption = 0
sigmaMg22 = not_a_number
sigmaMg21 = 0.288
test_mode = 0
produce_Plots = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        global_config.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "sigmaMgIIh" in msg  # key name used in _as_float
    assert "Invalid float" in msg

    assert any(
        "sigmaMgIIh" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )
def test_parse_simple_kv_ignores_lines_without_equals(tmp_path):
    cfg_path = tmp_path / "global_no_equals.cfg"
    cfg_path.write_text(
        """
line_core_emission = 1
this line has no equals sign
interstellar_absorption = 0
test_mode = 0
produce_Plots = 0
""",
        encoding="utf-8",
    )

    cfg = global_config.load_global_config(cfg_path)

    # The invalid line should simply be ignored, not cause errors
    assert cfg.line_core_emission is True
    assert cfg.interstellar_absorption is False
