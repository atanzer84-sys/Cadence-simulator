import pytest
from pathlib import Path

import configs.user_config as user_config
from configs.user_config import load_user_config, get_user_config


@pytest.fixture(autouse=True)
def reset_user_cache():
    user_config._USER = None
    yield
    user_config._USER = None


def _write_params(tmp_path, content: str) -> Path:
    p = tmp_path / "parameters.txt"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# CACHING + LOAD/GET
# ----------------------------------------------------------------------

# Tests: get_user_config
# Behavior: raises RuntimeError + logs ERROR when config not loaded
def test_get_user_config_raises_if_not_loaded(caplog):
    with pytest.raises(RuntimeError):
        get_user_config()

    assert any(
        "User config not loaded" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


# Tests: load_user_config
# Behavior: propagates errors from _read_user_cfg
def test_load_user_config_invalid_file_raises(monkeypatch):
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        raise ValueError("bad format")

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    with pytest.raises(ValueError, match="bad format"):
        load_user_config(Path("parameters.txt"))


# Tests: load_user_config
# Behavior: returns and caches whatever _read_user_cfg returns
def test_load_user_config_success(monkeypatch):
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        return {"ok": True}

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    result = load_user_config(Path("parameters.txt"))
    assert result == {"ok": True}


# ----------------------------------------------------------------------
# VALID FULL CONFIG
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# Behavior: parses all required fields correctly
def test_load_user_config_valid_full(tmp_path, make_user_cfg):
    content = make_user_cfg(
        target_name="HD 202772 A",
        total_observation_length_h="20.5",
        exposure_NUV_s="3",
        exposure_VIS_s="4.25",
        exposure_IR_s="10",
    )
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "HD 202772 A"
    assert cfg.total_observation_length_h == 20.5
    assert cfg.exposure_NUV_s == 3.0
    assert cfg.exposure_VIS_s == 4.25
    assert cfg.exposure_IR_s == 10.0


# ----------------------------------------------------------------------
# REQUIRED KEYS
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# Behavior: missing required key → ValueError with key name
@pytest.mark.parametrize("missing_key", [
    "target_name",
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_missing_required_raises(tmp_path, missing_key, make_user_cfg):
    base = make_user_cfg().splitlines()
    base = [line for line in base if not line.strip().startswith(f"{missing_key} =")]
    content = "\n".join(base)
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        load_user_config(path)

    assert missing_key in str(exc.value)


# Tests: _read_user_cfg
# Behavior: KeyError block produces ValueError containing only missing key
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
        load_user_config(path)

    assert "exposure_IR_s" in str(exc.value)


# ----------------------------------------------------------------------
# INVALID NUMERIC VALUES
# ----------------------------------------------------------------------

# Tests: _read_user_cfg
# Behavior: invalid numeric values raise ValueError containing key name
@pytest.mark.parametrize("bad_key", [
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_invalid_number_raises(tmp_path, bad_key, make_user_cfg):
    overrides = {
        "target_name": "Star",
        "total_observation_length_h": "1",
        "exposure_NUV_s": "1",
        "exposure_VIS_s": "1",
        "exposure_IR_s": "1",
    }
    overrides[bad_key] = "not_a_number"
    content = make_user_cfg(**overrides)
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        load_user_config(path)

    assert bad_key in str(exc.value)


# ----------------------------------------------------------------------
# MISSING FILE
# ----------------------------------------------------------------------

# Tests: load_user_config
# Behavior: missing file → FileNotFoundError + ERROR log
def test_user_config_missing_file_raises(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        load_user_config(missing)

    assert any(
        rec.levelname == "ERROR" and "does_not_exist.cfg" in rec.message
        for rec in caplog.records
    )
