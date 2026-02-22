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
    """Write parameter file to tmp_path and return its Path."""
    p = tmp_path / "parameters.txt"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def _write_min_params_with_target(tmp_path, target_name: str) -> Path:
    content = (
        f"target_name = {target_name}\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n"
    )
    return _write_params(tmp_path, content)


def test_get_user_config_raises_if_not_loaded(caplog):
    """get_user_config() raises RuntimeError with error log when config was never loaded."""
    with pytest.raises(RuntimeError) as exc:
        get_user_config()

    assert "User config not loaded" in str(exc.value)
    assert any(
        "User config not loaded" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_load_user_config_invalid_file_raises(monkeypatch, tmp_path):
    """Propagated errors from _read_user_cfg (e.g. bad format) are raised by load_user_config."""
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        raise ValueError("bad format")

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    with pytest.raises(ValueError, match="bad format"):
        user_config.load_user_config(Path("parameters.txt"))


def test_load_user_config_success(monkeypatch, tmp_path):
    """load_user_config returns and caches whatever _read_user_cfg returns (here mocked as dict)."""
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        return {"ok": True}

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    result = user_config.load_user_config(Path("parameters.txt"))
    assert result == {"ok": True}


# --- Valid inputs ---
def test_load_user_config_valid_full(tmp_path):
    """All required parameters parse correctly and are exposed on UserConfig."""
    content = """
    target_name = HD 202772 A
    total_observation_length_h = 20.5
    exposure_NUV_s = 3
    exposure_VIS_s = 4.25
    exposure_IR_s = 10
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "HD 202772 A"
    assert cfg.total_observation_length_h == 20.5
    assert cfg.exposure_NUV_s == 3.0
    assert cfg.exposure_VIS_s == 4.25
    assert cfg.exposure_IR_s == 10.0


def test_load_user_config_strips_quotes_double(tmp_path):
    """Double-quoted target_name is stripped of quotes."""
    content = """
    target_name = "HD 202772 A"
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "HD 202772 A"


def test_load_user_config_strips_quotes_single(tmp_path):
    """Single-quoted target_name is stripped of quotes."""
    content = """
    target_name = 'HD 202772 A'
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "HD 202772 A"


def test_load_user_config_ignores_comments_and_blank_lines(tmp_path):
    """Lines starting with # and blank lines are skipped by the parser."""
    content = """
    # comment
    target_name = Star
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "Star"


def test_load_user_config_handles_ugly_whitespace(tmp_path):
    """Whitespace around = and missing spaces is tolerated; values parse correctly."""
    content = """
target_name=HD 202772 A
total_observation_length_h  =  20.5
exposure_NUV_s=3
exposure_VIS_s = 4.25
exposure_IR_s  =  10
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == "HD 202772 A"
    assert cfg.total_observation_length_h == 20.5
    assert cfg.exposure_NUV_s == 3.0
    assert cfg.exposure_VIS_s == 4.25
    assert cfg.exposure_IR_s == 10.0


# --- Missing required parameters ---
@pytest.mark.parametrize("missing_key", [
    "target_name",
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_missing_required_raises(tmp_path, missing_key):
    """Missing any required parameter raises ValueError (KeyError from raw[] is converted)."""
    base = {
        "target_name": "HD 202772 A",
        "total_observation_length_h": "20.5",
        "exposure_NUV_s": "3",
        "exposure_VIS_s": "4.25",
        "exposure_IR_s": "10",
    }
    del base[missing_key]

    content = "\n".join(f"{k} = {v}" for k, v in base.items())
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError):
        load_user_config(path)


# --- Invalid types ---
@pytest.mark.parametrize("bad_key", [
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_user_config_invalid_number_raises(tmp_path, bad_key):
    """Non-numeric value for a float parameter raises ValueError from _as_float."""
    base = {
        "target_name": "Star",
        "total_observation_length_h": "1",
        "exposure_NUV_s": "1",
        "exposure_VIS_s": "1",
        "exposure_IR_s": "1",
    }
    base[bad_key] = "not_a_number"

    content = "\n".join(f"{k} = {v}" for k, v in base.items())
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError):
        load_user_config(path)


# --- Empty target_name ---
def test_load_user_config_empty_target_name_is_allowed(tmp_path):
    """Empty target_name is allowed and becomes '' after sanitization."""
    content = """
    target_name =
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == ""


@pytest.mark.parametrize(
    "target_name, expected",
    [
        ("HF 123", "HF 123"),
        (" HF 123 ", "HF 123"),
        ("'HF 123'", "HF 123"),
        ('"HF 123"', "HF 123"),
        ("' HF 123 '", "HF 123"),
        ("HF 123B", "HF 123B"),
    ],
)
def test_target_name_accepts_star_only_variants(tmp_path, target_name, expected):
    """target_name accepts star-only variants (spaces, quotes) and sanitizes to expected."""
    path = _write_min_params_with_target(tmp_path, target_name)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == expected


@pytest.mark.parametrize(
    "target_name, expected",
    [
        ("HF 123 b", "HF 123 b"),
        ("HF 123 B", "HF 123 B"),
        ("HF 123 b ", "HF 123 b"),
        ("'HF 123 b'", "HF 123 b"),
        ('"HF 123 b"', "HF 123 b"),
        ("' HF 123 b '", "HF 123 b"),
        ("HF   123", "HF   123"),
        ("'HF   123'", "HF   123"),
        ('HF    123', "HF    123"),
        ("'HF    123 '", "HF    123"),
    ],
)
def test_target_name_sanitizes_but_does_not_reject(tmp_path, target_name, expected):
    """target_name sanitizes whitespace and quotes but does not reject planet or multi-space names."""
    path = _write_min_params_with_target(tmp_path, target_name)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == expected


def test_user_config_missing_file_raises(tmp_path, caplog):
    """Loading a non-existent parameter file logs ERROR and raises FileNotFoundError."""
    user_config._USER = None  # force reload

    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        load_user_config(missing)

    assert any(
        "not found" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_missing_required_key_hits_keyerror_block(tmp_path):
    """Missing required key (e.g. exposure_IR_s) raises ValueError with the key name in the message."""
    content = """
    target_name = Star
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    # exposure_IR_s is missing
    """
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError) as exc:
        load_user_config(path)

    assert "exposure_IR_s" in str(exc.value)