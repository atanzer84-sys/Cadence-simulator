import pytest
from pathlib import Path

import configs.user_config as user_config


@pytest.fixture(autouse=True)
def reset_user_cache():
    user_config._USER = None
    yield
    user_config._USER = None


def _write_params(tmp_path, content: str) -> Path:
    p = tmp_path / "parameters.txt"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def _make_user_cfg_text(**overrides) -> str:
    base = dict(
        target_name="HD 202772 A",
        total_observation_length_h="20.5",
        exposure_NUV_s="3",
        exposure_VIS_s="4.25",
        exposure_IR_s="10",
    )
    base.update(overrides)
    return "\n".join(f"{k} = {v}" for k, v in base.items())


# ----------------------------------------------------------------------
# CACHING + LOAD/GET
# ----------------------------------------------------------------------

# Tests: get_user_config
# verifies that get_user_config raises RuntimeError when no config was loaded
# checks that an ERROR log entry is emitted in that case
def test_get_user_config_raises_if_not_loaded(caplog):
    with pytest.raises(RuntimeError):
        user_config.get_user_config()

    assert any(
        "User config not loaded" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


# Tests: load_user_config
# verifies that load_user_config propagates errors raised by _read_user_cfg
# checks that the loader does not swallow underlying parsing failures
def test_load_user_config_invalid_file_raises(monkeypatch):
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        raise ValueError("bad format")

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    with pytest.raises(ValueError, match="bad format"):
        user_config.load_user_config(Path("parameters.txt"))


# Tests: load_user_config
# verifies that load_user_config returns and caches the object from _read_user_cfg
# checks the cache wiring without depending on real file parsing
def test_load_user_config_success(monkeypatch):
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        return {"ok": True}

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    result = user_config.load_user_config(Path("parameters.txt"))
    assert result == {"ok": True}


# ----------------------------------------------------------------------
# VALID FULL CONFIG
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# verifies that a complete parameter file is parsed correctly
# checks that all required fields are converted to the expected types
def test_load_user_config_valid_full(tmp_path):
    content = _make_user_cfg_text(
        target_name="HD 202772 A",
        total_observation_length_h="20.5",
        exposure_NUV_s="3",
        exposure_VIS_s="4.25",
        exposure_IR_s="10",
    )
    path = _write_params(tmp_path, content)

    user_config.load_user_config(path)
    cfg = user_config.get_user_config()

    assert cfg.target_name == "HD 202772 A"
    assert cfg.total_observation_length_h == 20.5
    assert cfg.exposure_NUV_s == 3.0
    assert cfg.exposure_VIS_s == 4.25
    assert cfg.exposure_IR_s == 10.0


# ----------------------------------------------------------------------
# REQUIRED KEYS
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# verifies that removing any required key raises ValueError
# checks that the error message contains the missing key name
@pytest.mark.parametrize("missing_key", [
    "target_name",
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_missing_required_raises(tmp_path, missing_key):
    base = _make_user_cfg_text().splitlines()
    base = [line for line in base if not line.strip().startswith(f"{missing_key} =")]
    content = "\n".join(base)
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        user_config.load_user_config(path)

    assert missing_key in str(exc.value)


# Tests: _read_user_cfg
# verifies that the missing-key path hits the KeyError handling block
# checks that the resulting ValueError contains the missing parameter name
def test_missing_required_key_hits_keyerror_block(tmp_path):
    content = """
    target_name = Star
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    # exposure_IR_s missing
    """
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        user_config.load_user_config(path)

    assert "exposure_IR_s" in str(exc.value)


# ----------------------------------------------------------------------
# INVALID NUMERIC VALUES
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# verifies that invalid numeric values for required float fields raise ValueError
# checks that the error message names the field that failed conversion
@pytest.mark.parametrize("bad_key", [
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_invalid_number_raises(tmp_path, bad_key):
    overrides = {
        "target_name": "Star",
        "total_observation_length_h": "1",
        "exposure_NUV_s": "1",
        "exposure_VIS_s": "1",
        "exposure_IR_s": "1",
    }
    overrides[bad_key] = "not_a_number"
    content = _make_user_cfg_text(**overrides)
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        user_config.load_user_config(path)

    assert bad_key in str(exc.value)


# ----------------------------------------------------------------------
# MISSING FILE
# ----------------------------------------------------------------------

# Tests: load_user_config
# verifies that loading a missing parameter file raises FileNotFoundError
# checks that an ERROR log entry mentions the missing file
def test_user_config_missing_file_raises(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        user_config.load_user_config(missing)

    assert any(
        rec.levelname == "ERROR" and "does_not_exist.cfg" in rec.message
        for rec in caplog.records
    )