"""Tests for load_parameters with different parameter file contents."""

import pytest

from loaders.userparameter_loader import load_parameters


def _write_params(tmp_path, content: str):
    """Write content to a parameters file and return its path."""
    path = tmp_path / "parameters.txt"
    path.write_text(content.strip(), encoding="utf-8")
    return path


# --- Valid inputs ---
def test_load_parameters_valid_full(tmp_path):
    content = """
    target_name = HD 202772 A
    total_observation_length_h = 20.5
    exposure_NUV_s = 3
    exposure_VIS_s = 4.25
    exposure_IR_s = 10
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "HD 202772 A"
    assert params["total_observation_length_h"] == 20.5
    assert params["exposure_NUV_s"] == 3.0
    assert params["exposure_VIS_s"] == 4.25
    assert params["exposure_IR_s"] == 10.0


def test_load_parameters_strips_quotes_around_target_name(tmp_path):
    content = """
    target_name = "HD 202772 A"
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "HD 202772 A"


def test_load_parameters_strips_single_quotes_around_target_name(tmp_path):
    content = """
    target_name = 'HD 202772 A'
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "HD 202772 A"


def test_load_parameters_strips_mixed_quotes_around_target_name(tmp_path):
    content = """
    target_name = 'HD 202772 A"
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "HD 202772 A"


def test_load_parameters_ignores_comments_and_blank_lines(tmp_path):
    content = """
    # comment
    target_name = Star
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "Star"


def test_load_parameters_handles_ugly_whitespace(tmp_path):
    """No space around =, extra spaces around = and values; strip logic still yields correct values."""
    content = """
target_name=HD 202772 A
total_observation_length_h  =  20.5
exposure_NUV_s=3
exposure_VIS_s = 4.25
exposure_IR_s  =  10
    """
    path = _write_params(tmp_path, content)
    params = load_parameters(path)
    assert params["target_name"] == "HD 202772 A"
    assert params["total_observation_length_h"] == 20.5
    assert params["exposure_NUV_s"] == 3.0
    assert params["exposure_VIS_s"] == 4.25
    assert params["exposure_IR_s"] == 10.0


# --- Missing required parameters ---
@pytest.mark.parametrize("missing_key", [
    "target_name",
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_parameters_missing_required_raises(tmp_path, missing_key):
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
    with pytest.raises(ValueError, match=f"Missing required parameter: {missing_key}"):
        load_parameters(path)


# --- Invalid types (non-numeric where number required) ---
@pytest.mark.parametrize("bad_key", [
    "total_observation_length_h",
    "exposure_NUV_s",
    "exposure_VIS_s",
    "exposure_IR_s",
])
def test_load_parameters_invalid_number_raises(tmp_path, bad_key):
    """Any numeric param set to non-numeric raises ValueError."""
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
    with pytest.raises(ValueError, match=f"{bad_key} must be a number"):
        load_parameters(path)


# --- Empty target_name ---
def test_load_parameters_empty_target_name_raises(tmp_path):
    content = """
    target_name =
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    with pytest.raises(ValueError, match="target_name must not be empty"):
        load_parameters(path)


def test_load_parameters_target_name_only_quotes_raises(tmp_path):
    content = """
    target_name = "   "
    total_observation_length_h = 1
    exposure_NUV_s = 1
    exposure_VIS_s = 1
    exposure_IR_s = 1
    """
    path = _write_params(tmp_path, content)
    with pytest.raises(ValueError, match="target_name must not be empty"):
        load_parameters(path)
