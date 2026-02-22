import pytest
from pathlib import Path

import configs.global_config as gc

_CONFIG_DIR = Path(__file__).resolve().parent


def _cfg_path(name: str) -> Path:
    """Path to a config file in tests/configs/."""
    return _CONFIG_DIR / name


def _base_cfg(name: str) -> str:
    """Content of a config file in tests/configs/. Single source of truth; add/rename params only in that file."""
    return _cfg_path(name).read_text(encoding="utf-8")


def _cfg_drop_key(content: str, key: str) -> str:
    """Remove the line that assigns key (e.g. 'sigmaMg22') so loader uses default. Refactor-safe: only the key name is specified."""
    return "\n".join(
        line for line in content.splitlines()
        if not line.strip().startswith(key + " =")
    )


def _cfg_replace(content: str, old_substring: str, new_substring: str) -> str:
    """Replace one assignment in config content (e.g. to inject an invalid value). Refactor-safe: only the changed part is in the test."""
    return content.replace(old_substring, new_substring, 1)


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
    content = _cfg_drop_key(_base_cfg("global_full.cfg"), "sigmaMg22")
    cfg_path = tmp_path / "global_missing_sigma_h.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = gc.load_global_config(cfg_path)

    assert cfg.sigmaMg22 == gc.DEFAULT_SIGMA_MG22
    assert any("sigmaMg22 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_missing_sigmaMg21_uses_default_and_logs_warning(caplog, tmp_path):
    """Missing sigmaMg21 uses DEFAULT_SIGMA_MG21 and a WARNING is logged."""
    content = _cfg_drop_key(_base_cfg("global_full.cfg"), "sigmaMg21")
    cfg_path = tmp_path / "global_missing_sigma_k.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = gc.load_global_config(cfg_path)

    assert cfg.sigmaMg21 == gc.DEFAULT_SIGMA_MG21
    assert any("sigmaMg21 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_optional_float_parses_numeric():
    """Optional float fields parse numeric values correctly (e.g. mg2_col, mg1_col, fe2_col)."""
    cfg = gc.load_global_config(_cfg_path("global_full.cfg"))

    assert cfg.mg2_col == 12.5
    assert cfg.mg1_col == 3.0
    assert cfg.fe2_col == 0.001


def test_optional_float_invalid_value_raises(tmp_path):
    """Invalid value for an optional float (e.g. mg2_col) raises ValueError."""
    content = _cfg_replace(_base_cfg("global_full.cfg"), "mg2_col = 12.5", "mg2_col = not_a_number")
    cfg_path = tmp_path / "global_invalid_optional.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        gc.load_global_config(cfg_path)


def test_invalid_float_reports_property_name(caplog, tmp_path):
    """Invalid float value raises ValueError and logs ERROR with the key name used in code (e.g. sigmaMgIIh)."""
    content = _cfg_replace(_base_cfg("global_full.cfg"), "sigmaMg22 = 0.257", "sigmaMg22 = not_a_number")
    cfg_path = tmp_path / "global_invalid_float.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "sigmaMgIIh" in msg  # key name used in _as_float
    assert "Invalid float" in msg

    assert any(
        "sigmaMgIIh" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_invalid_sigmaMg21_reports_property_name(caplog, tmp_path):
    """Invalid sigmaMg21 value raises ValueError and logs ERROR with the key name used in code (sigmaMgIIk)."""
    content = _cfg_replace(_base_cfg("global_full.cfg"), "sigmaMg21 = 0.288", "sigmaMg21 = not_a_number")
    cfg_path = tmp_path / "global_invalid_sigma_k.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "sigmaMgIIk" in msg
    assert "Invalid float" in msg

    assert any(
        "sigmaMgIIk" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_int_fields_parse_numeric():
    """Int fields (n_bias_and_darkframes, n_science_frames_per_channel) parse numeric values correctly."""
    cfg = gc.load_global_config(_cfg_path("global_full.cfg"))

    assert cfg.n_bias_and_darkframes == 3
    assert cfg.n_science_frames_per_channel == 7


def test_invalid_int_reports_property_name(caplog, tmp_path):
    """Invalid value for an int field (e.g. n_bias_and_darkframes) raises ValueError and logs ERROR with the key name."""
    content = _cfg_replace(_base_cfg("global_full.cfg"), "n_bias_and_darkframes = 3", "n_bias_and_darkframes = not_an_int")
    cfg_path = tmp_path / "global_invalid_int.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "n_bias_and_darkframes" in msg
    assert "Invalid int" in msg

    assert any(
        "n_bias_and_darkframes" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_parse_simple_kv_ignores_lines_without_equals(tmp_path):
    """Lines without '=' are ignored by the parser; valid key=value lines are still parsed."""
    content = _cfg_replace(_base_cfg("global_full.cfg"), "line_core_emission = 0", "line_core_emission = 1")
    content = content.replace("interstellar_absorption = 0\n", "interstellar_absorption = 0\nthis line has no equals sign\n")
    cfg_path = tmp_path / "global_no_equals.cfg"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = gc.load_global_config(cfg_path)

    assert cfg.line_core_emission is True
    assert cfg.interstellar_absorption is False


def test_required_log_r_fields_reject_non_numeric(tmp_path):
    """Required log_r fields (log_r_teff_threshold, log_r_hot_value, log_r_cool_value) reject non-numeric values with ValueError."""
    base = _base_cfg("global_full.cfg")

    # 1) bad teff threshold
    cfg_path = tmp_path / "bad_teff.cfg"
    cfg_path.write_text(_cfg_replace(base, "log_r_teff_threshold = 5500", "log_r_teff_threshold = banana"), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_teff_threshold" in str(exc.value)

    # 2) bad hot value
    cfg_path = tmp_path / "bad_hot.cfg"
    cfg_path.write_text(_cfg_replace(base, "log_r_hot_value = -4.2", "log_r_hot_value = banana"), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_hot_value" in str(exc.value)

    # 3) bad cool value
    cfg_path = tmp_path / "bad_cool.cfg"
    cfg_path.write_text(_cfg_replace(base, "log_r_cool_value = -4.8", "log_r_cool_value = banana"), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_cool_value" in str(exc.value)


def test_required_log_r_fields_parse_numeric():
    """Required log_r fields parse numeric values correctly (enable_log_r_fallback, threshold, hot/cool values)."""
    cfg = gc.load_global_config(_cfg_path("global_full.cfg"))

    assert cfg.enable_log_r_fallback is True
    assert cfg.log_r_teff_threshold == 5500.0
    assert cfg.log_r_hot_value == -4.2
    assert cfg.log_r_cool_value == -4.8


def test_optional_bool_and_int_fields_omitted_use_defaults():
    """When write_dark_and_bias_png, write_science_frames_png, produce_Plots and int frame fields are omitted, they use default False/0."""
    cfg = gc.load_global_config(_cfg_path("global_minimal.cfg"))

    assert cfg.write_dark_and_bias_png is False
    assert cfg.write_science_frames_png is False
    assert cfg.produce_Plots is False
    assert cfg.n_bias_and_darkframes == 0
    assert cfg.n_science_frames_per_channel == 0


def test_optional_bool_and_int_fields_set_parsed_correctly():
    """When write_dark_and_bias_png, write_science_frames_png, produce_Plots and int frame fields are set, they are parsed correctly."""
    cfg = gc.load_global_config(_cfg_path("global_full.cfg"))

    assert cfg.produce_Plots is True
    assert cfg.write_dark_and_bias_png is True
    assert cfg.write_science_frames_png is True
    assert cfg.n_bias_and_darkframes == 3
    assert cfg.n_science_frames_per_channel == 7