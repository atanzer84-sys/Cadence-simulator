import pytest
import configs.user_config as user_config
from configs.user_config import load_user_config, get_user_config

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


def test_load_user_config_strips_mixed_quotes(tmp_path):
    content = """
    target_name = 'HD 202772 A"
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

    with pytest.raises(KeyError):
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
def test_load_user_config_empty_target_name_raises(tmp_path):
    content = """
    target_name =
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError):
        load_user_config(path)


def test_load_user_config_target_name_only_quotes_raises(tmp_path):
    content = """
    target_name = "   "
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)

    with pytest.raises(ValueError):
        load_user_config(path)


# --- Target name sanitization ---
def test_target_name_with_planet_letter_is_preserved(tmp_path):
    p = tmp_path / "parameters.txt"
    p.write_text(
        "target_name = HF 123 b\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n"
    )

    load_user_config(p)
    cfg = get_user_config()

    assert cfg.target_name == "HF 123 b"


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
