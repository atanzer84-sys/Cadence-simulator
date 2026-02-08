import pytest
import configs.user_config as user_config
from configs.user_config import load_user_config, get_user_config
from pathlib import Path

@pytest.fixture(autouse=True)
def reset_user_cache():
    user_config._USER = None
    yield
    user_config._USER = None



def _write_params(tmp_path, content: str) -> str:
    p = tmp_path / "parameters.txt"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def _write_min_params_with_target(tmp_path, target_name: str) -> str:
    content = (
        f"target_name = {target_name}\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n"
    )
    return _write_params(tmp_path, content)

def test_load_user_config_invalid_file_raises(monkeypatch, tmp_path):
    # ensure no cache pollution
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        raise ValueError("bad format")

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    with pytest.raises(ValueError, match="bad format"):
        user_config.load_user_config(Path("parameters.txt"))

def test_load_user_config_success(monkeypatch, tmp_path):
    monkeypatch.setattr(user_config, "_USER", None)

    def fake_read(_path: Path):
        return {"ok": True}

    monkeypatch.setattr(user_config, "_read_user_cfg", fake_read)

    result = user_config.load_user_config(Path("parameters.txt"))
    assert result == {"ok": True}

# --- Valid inputs ---
def test_load_user_config_valid_full(tmp_path):
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

    # empty target name should now be accepted
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
    path = _write_min_params_with_target(tmp_path, target_name)

    load_user_config(path)
    cfg = get_user_config()

    assert cfg.target_name == expected

def test_user_config_missing_file_raises(tmp_path, caplog):
    user_config._USER = None  # force reload

    missing = tmp_path / "does_not_exist.cfg"

    with pytest.raises(FileNotFoundError):
        load_user_config(missing)

    assert any(
        "not found" in rec.message.lower() and rec.levelname == "ERROR"
        for rec in caplog.records
    )
def test_missing_required_key_hits_keyerror_block(tmp_path):
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
