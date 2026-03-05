import pytest
from pathlib import Path

import configs.global_config as gc

_CONFIG_DIR = Path(__file__).resolve().parent


def _full_cfg_path() -> Path:
    """Path to the single canonical config file in tests/configs/."""
    return _CONFIG_DIR / "global_full.cfg"


def _base_cfg() -> str:
    """Canonical config content used as baseline for all global-config tests."""
    return _full_cfg_path().read_text(encoding="utf-8")


def _cfg_drop_key(content: str, key: str) -> str:
    """Remove the line that assigns key (e.g. 'sigmaMg22') so loader uses default. Refactor-safe: only the key name is specified."""
    return "\n".join(
        line for line in content.splitlines()
        if not line.strip().startswith(key + " =")
    )


def _cfg_replace(content: str, old_substring: str, new_substring: str) -> str:
    """Replace one assignment in config content (e.g. to inject an invalid value). Refactor-safe: only the changed part is in the test."""
    return content.replace(old_substring, new_substring, 1)


def _cfg_write(tmp_path: Path, filename: str, content: str) -> Path:
    """Write config content to a temp file and return the path."""
    cfg_path = tmp_path / filename
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


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
    cfg = gc.load_global_config(_full_cfg_path())
    assert isinstance(cfg, gc.GlobalConfig)


def test_load_global_config_is_cached_and_shared():
    """Loading the same path twice returns the same cached instance; get_global_config() shares it."""
    cfg1 = gc.load_global_config(_full_cfg_path())
    cfg2 = gc.load_global_config(_full_cfg_path())

    assert cfg1 is cfg2
    assert gc.get_global_config() is cfg1


def test_optional_float_fields_accept_blank_and_none(tmp_path):
    """Optional float fields (mg2_col, mg1_col, fe2_col) accept blank or none and become None."""
    content = _base_cfg()
    content = _cfg_replace(content, "mg2_col = 12.5", "mg2_col = ")
    content = _cfg_replace(content, "mg1_col = 3", "mg1_col = none")
    content = _cfg_replace(content, "fe2_col = 0.001", "fe2_col = ")
    cfg_path = _cfg_write(tmp_path, "global_optional_blanks.cfg", content)
    cfg = gc.load_global_config(cfg_path)

    assert cfg.mg2_col is None
    assert cfg.mg1_col is None
    assert cfg.fe2_col is None


def test_boolean_fields_accept_common_spellings(tmp_path):
    """Boolean fields accept common spellings (true/false, yes/no, 0/1)."""
    content = _base_cfg()
    content = _cfg_replace(content, "line_core_emission = 0", "line_core_emission = 0")
    content = _cfg_replace(content, "interstellar_absorption = 0", "interstellar_absorption = yes")
    content = _cfg_replace(content, "test_mode = 0", "test_mode = off")
    cfg_path = _cfg_write(tmp_path, "global_boolean_spellings.cfg", content)
    cfg = gc.load_global_config(cfg_path)

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


def test_invalid_boolean_reports_property_name(caplog, tmp_path):
    """Invalid boolean value raises ValueError and logs ERROR with the config key name."""
    content = _cfg_replace(_base_cfg(), "line_core_emission = 0", "line_core_emission = maybe")
    cfg_path = _cfg_write(tmp_path, "global_invalid_bool.cfg", content)
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "line_core_emission" in msg
    assert "Invalid boolean value" in msg

    assert any(
        "line_core_emission" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_missing_sigmaMg22_uses_default_and_logs_warning(caplog, tmp_path):
    """Missing sigmaMg22 uses DEFAULT_SIGMA_MG22 and a WARNING is logged."""
    content = _cfg_drop_key(_base_cfg(), "sigmaMg22")
    cfg_path = _cfg_write(tmp_path, "global_missing_sigma_h.cfg", content)

    cfg = gc.load_global_config(cfg_path)

    assert cfg.sigmaMg22 == gc.DEFAULT_SIGMA_MG22
    assert any("sigmaMg22 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_missing_sigmaMg21_uses_default_and_logs_warning(caplog, tmp_path):
    """Missing sigmaMg21 uses DEFAULT_SIGMA_MG21 and a WARNING is logged."""
    content = _cfg_drop_key(_base_cfg(), "sigmaMg21")
    cfg_path = _cfg_write(tmp_path, "global_missing_sigma_k.cfg", content)

    cfg = gc.load_global_config(cfg_path)

    assert cfg.sigmaMg21 == gc.DEFAULT_SIGMA_MG21
    assert any("sigmaMg21 not provided" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)


def test_optional_float_parses_numeric():
    """Optional float fields parse numeric values correctly (e.g. mg2_col, mg1_col, fe2_col)."""
    cfg = gc.load_global_config(_full_cfg_path())

    assert cfg.mg2_col == 12.5
    assert cfg.mg1_col == 3.0
    assert cfg.fe2_col == 0.001


def test_optional_float_invalid_value_raises(tmp_path):
    """Invalid value for an optional float (e.g. mg2_col) raises ValueError."""
    content = _cfg_replace(_base_cfg(), "mg2_col = 12.5", "mg2_col = not_a_number")
    cfg_path = _cfg_write(tmp_path, "global_invalid_optional.cfg", content)

    with pytest.raises(ValueError):
        gc.load_global_config(cfg_path)


def test_invalid_float_reports_property_name(caplog, tmp_path):
    """Invalid float value raises ValueError and logs ERROR with the key name used in code (e.g. sigmaMgIIh)."""
    content = _cfg_replace(_base_cfg(), "sigmaMg22 = 0.257", "sigmaMg22 = not_a_number")
    cfg_path = _cfg_write(tmp_path, "global_invalid_float.cfg", content)

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
    content = _cfg_replace(_base_cfg(), "sigmaMg21 = 0.288", "sigmaMg21 = not_a_number")
    cfg_path = _cfg_write(tmp_path, "global_invalid_sigma_k.cfg", content)

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
    """Int fields (n_non_science_frames) parse numeric values correctly."""
    cfg = gc.load_global_config(_full_cfg_path())

    assert cfg.n_non_science_frames == 3


def test_invalid_int_reports_property_name(caplog, tmp_path):
    """Invalid value for an int field (e.g. n_non_science_frames) raises ValueError and logs ERROR with the key name."""
    content = _cfg_replace(_base_cfg(), "n_non_science_frames = 3", "n_non_science_frames = not_an_int")
    cfg_path = _cfg_write(tmp_path, "global_invalid_int.cfg", content)

    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)

    msg = str(exc.value)
    assert "n_non_science_frames" in msg
    assert "Invalid int" in msg

    assert any(
        "n_non_science_frames" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_parse_simple_kv_ignores_lines_without_equals(tmp_path):
    """Lines without '=' are ignored by the parser; valid key=value lines are still parsed."""
    content = _cfg_replace(_base_cfg(), "line_core_emission = 0", "line_core_emission = 1")
    content = content.replace("interstellar_absorption = 0\n", "interstellar_absorption = 0\nthis line has no equals sign\n")
    cfg_path = _cfg_write(tmp_path, "global_no_equals.cfg", content)

    cfg = gc.load_global_config(cfg_path)

    assert cfg.line_core_emission is True
    assert cfg.interstellar_absorption is False


def test_required_log_r_fields_reject_non_numeric(tmp_path):
    """Required log_r fields (log_r_teff_threshold, log_r_hot_value, log_r_cool_value) reject non-numeric values with ValueError."""
    base = _base_cfg()

    # 1) bad teff threshold
    cfg_path = _cfg_write(tmp_path, "bad_teff.cfg", _cfg_replace(base, "log_r_teff_threshold = 5500", "log_r_teff_threshold = banana"))
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_teff_threshold" in str(exc.value)

    # 2) bad hot value
    cfg_path = _cfg_write(tmp_path, "bad_hot.cfg", _cfg_replace(base, "log_r_hot_value = -4.8", "log_r_hot_value = banana"))
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_hot_value" in str(exc.value)

    # 3) bad cool value
    cfg_path = _cfg_write(tmp_path, "bad_cool.cfg", _cfg_replace(base, "log_r_cool_value = -4.2", "log_r_cool_value = banana"))
    with pytest.raises(ValueError) as exc:
        gc.load_global_config(cfg_path)
    assert "log_r_cool_value" in str(exc.value)


def test_required_log_r_fields_parse_numeric():
    """Required log_r fields parse numeric values correctly (enable_log_r_fallback, threshold, hot/cool values)."""
    cfg = gc.load_global_config(_full_cfg_path())

    assert cfg.enable_log_r_fallback is True
    assert cfg.log_r_teff_threshold == 5500.0
    assert cfg.log_r_hot_value == -4.8
    assert cfg.log_r_cool_value == -4.2


def test_optional_bool_and_int_fields_omitted_use_defaults(tmp_path):
    """When write_non_science_frames_png, write_science_frames_png, produce_plots and int frame fields are omitted, they use default False/0."""
    content = _base_cfg()
    for key in (
        "write_non_science_frames_png",
        "write_science_frames_png",
        "n_non_science_frames",
        "produce_plots",
    ):
        content = _cfg_drop_key(content, key)
    cfg_path = _cfg_write(tmp_path, "global_omitted_optional_bool_int.cfg", content)
    cfg = gc.load_global_config(cfg_path)

    assert cfg.write_non_science_frames_png is False
    assert cfg.write_science_frames_png is False
    assert cfg.produce_plots is False
    assert cfg.n_non_science_frames == 0


def test_optional_bool_and_int_fields_set_parsed_correctly():
    """When write_non_science_frames_png, write_science_frames_png, produce_plots and int frame fields are set, they are parsed correctly."""
    cfg = gc.load_global_config(_full_cfg_path())

    assert cfg.produce_plots is True
    assert cfg.write_non_science_frames_png is True
    assert cfg.write_science_frames_png is True
    assert cfg.n_non_science_frames == 3


def test_background_fields_parse_explicit_values():
    """Background-related fields parse explicit values from config."""
    cfg = gc.load_global_config(_full_cfg_path())

    assert cfg.background_type == "default"
    assert cfg.background_file == "background_default.txt"
    assert cfg.sky_pixel_area_arcsec2 == 25.0
    assert cfg.zod_dist_file == "zod_dist.fits"
    assert cfg.zod_spectrum_file == "zod_spectrum.txt"


def test_background_fields_accept_blank_or_none(tmp_path):
    """Blank/none values for optional background fields parse as None."""
    content = _base_cfg()
    content = _cfg_replace(content, "background_type = DEFAULT", "background_type = ")
    content = _cfg_replace(content, "background_file = background_default.txt", "background_file = none")
    content = _cfg_replace(content, "sky_pixel_area_arcsec2 = 25.0", "sky_pixel_area_arcsec2 = none")
    content = _cfg_replace(content, "zod_dist_file = zod_dist.fits", "zod_dist_file = ")
    content = _cfg_replace(content, "zod_spectrum_file = zod_spectrum.txt", "zod_spectrum_file = none")
    cfg_path = _cfg_write(tmp_path, "global_background_optional_none.cfg", content)

    cfg = gc.load_global_config(cfg_path)

    assert cfg.background_type is None
    assert cfg.background_file is None
    assert cfg.sky_pixel_area_arcsec2 is None
    assert cfg.zod_dist_file is None
    assert cfg.zod_spectrum_file is None


def test_background_type_is_lowercased():
    """background_type uses lowercase normalization when provided."""
    cfg = gc.load_global_config(_full_cfg_path())
    assert cfg.background_type == "default"


def test_invalid_sky_pixel_area_arcsec2_raises(tmp_path):
    """Invalid optional float for sky_pixel_area_arcsec2 raises ValueError."""
    content = _cfg_replace(
        _base_cfg(),
        "sky_pixel_area_arcsec2 = 25.0",
        "sky_pixel_area_arcsec2 = not_a_number",
    )
    cfg_path = _cfg_write(tmp_path, "global_invalid_sky_pixel_area.cfg", content)

    with pytest.raises(ValueError):
        gc.load_global_config(cfg_path)