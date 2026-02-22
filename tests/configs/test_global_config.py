import pytest
from pathlib import Path

import configs.global_config as gc

_CONFIG_DIR = Path(__file__).resolve().parent


def _cfg_path(name: str) -> Path:
    """Path to a config file in tests/configs/."""
    return _CONFIG_DIR / name


@pytest.fixture(autouse=True)
def reset_cache():
    gc._GLOBAL = None
    yield
    gc._GLOBAL = None


def test_get_global_config_raises_if_not_loaded(caplog):
    """get_global_config() raises RuntimeError with error log when config was never loaded."""
    with pytest.raises(RuntimeError) as exc:
        gc.get_global_config()

    assert "Global config not loaded" in str(exc.value)
    assert any(
        "Global config not loaded" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_load_global_config_returns_globalconfig_instance():
    """load_global_config() returns an instance of GlobalConfig."""
    cfg = gc.load_global_config(_cfg_path("global_minimal.cfg"))
    assert isinstance(cfg, gc.GlobalConfig)


def test_load_global_config_is_cached_and_shared():
    """Loading the same path twice returns the same cached instance; get_global_config() shares it."""
    cfg1 = gc.load_global_config(_cfg_path("global_minimal.cfg"))
    cfg2 = gc.load_global_config(_cfg_path("global_minimal.cfg"))

    assert cfg1 is cfg2
    assert gc.get_global_config() is cfg1


def test_optional_float_fields_accept_blank_and_none():
    """Optional float fields (mg2_col, mg1_col, fe2_col) accept blank or none and become None."""
    cfg = gc.load_global_config(_cfg_path("global_optional_blanks.cfg"))

    assert cfg.mg2_col is None
    assert cfg.mg1_col is None
    assert cfg.fe2_col is None


def test_boolean_fields_accept_common_spellings():
    """Boolean fields accept common spellings (true/false, yes/no, 0/1)."""
    cfg = gc.load_global_config(_cfg_path("global_optional_blanks.cfg"))

    assert cfg.line_core_emission is False
    assert cfg.interstellar_absorption is True
    assert cfg.test_mode is False


def test_missing_config_file_logs_and_raises(caplog, tmp_path):
    """Loading a non-existent config file logs ERROR and raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        gc.load_global_config(missing)

    assert any(
        "not found" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_invalid_boolean_reports_property_name(caplog):
    """Invalid boolean value raises ValueError and logs ERROR with the config key name."""
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(_cfg_path("global_invalid_bool.cfg"))

    msg = str(exc.value)
    assert "line_core_emission" in msg
    assert "Invalid boolean value" in msg

    assert any(
        "line_core_emission" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_missing_sigmaMg22_uses_default_and_logs_warning(caplog, tmp_path):
    """Missing sigmaMg22 uses DEFAULT_SIGMA_MG22 and a WARNING is logged."""
    cfg_path = tmp_path / "global_missing_sigma_h.cfg"
    cfg_path.write_text(
        """
line_core_emission = 1
interstellar_absorption = 0
sigmaMg21 = 0.288
test_mode = 0
enable_log_r_fallback = 0
log_r_teff_threshold = 5500
log_r_hot_value = 0.0
log_r_cool_value = 0.0
""",
        encoding="utf-8",
    )

    cfg = gc.load_global_config(cfg_path)

    assert cfg.sigmaMg22 == gc.DEFAULT_SIGMA_MG22
    assert any("sigmaMg22 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_optional_float_parses_numeric(tmp_path):
    """Optional float fields parse numeric values correctly (e.g. mg2_col, mg1_col, fe2_col)."""
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
enable_log_r_fallback = 0
log_r_teff_threshold = 5500
log_r_hot_value = 0.0
log_r_cool_value = 0.0
""",
        encoding="utf-8",
    )

    cfg = gc.load_global_config(cfg_path)

    assert cfg.mg2_col == 12.5
    assert cfg.mg1_col == 3.0
    assert cfg.fe2_col == 0.001


def test_optional_float_invalid_value_raises(tmp_path):
    """Invalid value for an optional float (e.g. mg2_col) raises ValueError."""
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
        gc.load_global_config(cfg_path)


def test_invalid_float_reports_property_name(caplog, tmp_path):
    """Invalid float value raises ValueError and logs ERROR with the key name used in code (e.g. sigmaMgIIh)."""
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
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "sigmaMgIIh" in msg  # key name used in _as_float
    assert "Invalid float" in msg

    assert any(
        "sigmaMgIIh" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_parse_simple_kv_ignores_lines_without_equals(tmp_path):
    """Lines without '=' are ignored by the parser; valid key=value lines are still parsed."""
    cfg_path = tmp_path / "global_no_equals.cfg"
    cfg_path.write_text(
        """
line_core_emission = 1
this line has no equals sign
interstellar_absorption = 0
test_mode = 0
produce_Plots = 0
enable_log_r_fallback = 0
log_r_teff_threshold = 5500
log_r_hot_value = 0.0
log_r_cool_value = 0.0
""",
        encoding="utf-8",
    )

    cfg = gc.load_global_config(cfg_path)

    # The invalid line should simply be ignored, not cause errors
    assert cfg.line_core_emission is True
    assert cfg.interstellar_absorption is False


def test_required_log_r_fields_reject_non_numeric(tmp_path):
    """Required log_r fields (log_r_teff_threshold, log_r_hot_value, log_r_cool_value) reject non-numeric values with ValueError."""
    base_cfg = """
line_core_emission = 0
interstellar_absorption = 0
enable_log_r_fallback = 1
log_r_teff_threshold = 5500
log_r_hot_value = -4.2
log_r_cool_value = -4.8
test_mode = 0
produce_Plots = 0
"""

    # 1) bad teff threshold
    cfg_path = tmp_path / "bad_teff.cfg"
    cfg_path.write_text(
        base_cfg.replace("log_r_teff_threshold = 5500", "log_r_teff_threshold = banana"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_teff_threshold" in str(exc.value)

    # 2) bad hot value
    cfg_path = tmp_path / "bad_hot.cfg"
    cfg_path.write_text(
        base_cfg.replace("log_r_hot_value = -4.2", "log_r_hot_value = banana"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_hot_value" in str(exc.value)

    # 3) bad cool value
    cfg_path = tmp_path / "bad_cool.cfg"
    cfg_path.write_text(
        base_cfg.replace("log_r_cool_value = -4.8", "log_r_cool_value = banana"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_cool_value" in str(exc.value)


def test_required_log_r_fields_parse_numeric(tmp_path):
    """Required log_r fields parse numeric values correctly (enable_log_r_fallback, threshold, hot/cool values)."""
    cfg_path = tmp_path / "global_required_log_r_numeric.cfg"
    cfg_path.write_text(
        """
line_core_emission = 0
interstellar_absorption = 0
enable_log_r_fallback = 1
log_r_teff_threshold = 5500
log_r_hot_value = -4.2
log_r_cool_value = -4.8
test_mode = 0
produce_Plots = 0
""",
        encoding="utf-8",
    )

    cfg = gc.load_global_config(cfg_path)

    assert cfg.enable_log_r_fallback is True
    assert cfg.log_r_teff_threshold == 5500.0
    assert cfg.log_r_hot_value == -4.2
    assert cfg.log_r_cool_value == -4.8