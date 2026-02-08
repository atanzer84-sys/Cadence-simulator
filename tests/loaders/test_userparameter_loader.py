# """Tests for load_parameters with different parameter file contents."""

# import pytest

# from loaders.userparameter_loader import load_parameters


# def _write_params(tmp_path, content: str) -> str:
#     p = tmp_path / "parameters.txt"
#     p.write_text(content.strip() + "\n", encoding="utf-8")
#     return str(p)

# def _write_min_params_with_target(tmp_path, target_name: str) -> str:
#     content = (
#         f"target_name = {target_name}\n"
#         "total_observation_length_h = 1\n"
#         "exposure_NUV_s = 1\n"
#         "exposure_VIS_s = 1\n"
#         "exposure_IR_s = 1\n"
#     )
#     return _write_params(tmp_path, content)


# # --- Valid inputs ---
# def test_load_parameters_valid_full(tmp_path):
#     content = """
#     target_name = HD 202772 A
#     total_observation_length_h = 20.5
#     exposure_NUV_s = 3
#     exposure_VIS_s = 4.25
#     exposure_IR_s = 10
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "HD 202772 A"
#     assert params["total_observation_length_h"] == 20.5
#     assert params["exposure_NUV_s"] == 3.0
#     assert params["exposure_VIS_s"] == 4.25
#     assert params["exposure_IR_s"] == 10.0


# def test_load_parameters_strips_quotes_around_target_name(tmp_path):
#     content = """
#     target_name = "HD 202772 A"
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "HD 202772 A"


# def test_load_parameters_strips_single_quotes_around_target_name(tmp_path):
#     content = """
#     target_name = 'HD 202772 A'
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "HD 202772 A"


# def test_load_parameters_strips_mixed_quotes_around_target_name(tmp_path):
#     content = """
#     target_name = 'HD 202772 A"
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "HD 202772 A"


# def test_load_parameters_ignores_comments_and_blank_lines(tmp_path):
#     content = """
#     # comment
#     target_name = Star
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "Star"


# def test_load_parameters_handles_ugly_whitespace(tmp_path):
#     """No space around =, extra spaces around = and values; strip logic still yields correct values."""
#     content = """
# target_name=HD 202772 A
# total_observation_length_h  =  20.5
# exposure_NUV_s=3
# exposure_VIS_s = 4.25
# exposure_IR_s  =  10
#     """
#     path = _write_params(tmp_path, content)
#     params = load_parameters(path)
#     assert params["target_name"] == "HD 202772 A"
#     assert params["total_observation_length_h"] == 20.5
#     assert params["exposure_NUV_s"] == 3.0
#     assert params["exposure_VIS_s"] == 4.25
#     assert params["exposure_IR_s"] == 10.0


# # --- Missing required parameters ---
# @pytest.mark.parametrize("missing_key", [
#     "target_name",
#     "total_observation_length_h",
#     "exposure_NUV_s",
#     "exposure_VIS_s",
#     "exposure_IR_s",
# ])
# def test_load_parameters_missing_required_raises(tmp_path, missing_key):
#     base = {
#         "target_name": "HD 202772 A",
#         "total_observation_length_h": "20.5",
#         "exposure_NUV_s": "3",
#         "exposure_VIS_s": "4.25",
#         "exposure_IR_s": "10",
#     }
#     del base[missing_key]
#     content = "\n".join(f"{k} = {v}" for k, v in base.items())
#     path = _write_params(tmp_path, content)
#     with pytest.raises(ValueError, match=f"Missing required parameter: {missing_key}"):
#         load_parameters(path)


# # --- Invalid types (non-numeric where number required) ---
# @pytest.mark.parametrize("bad_key", [
#     "total_observation_length_h",
#     "exposure_NUV_s",
#     "exposure_VIS_s",
#     "exposure_IR_s",
# ])
# def test_load_parameters_invalid_number_raises(tmp_path, bad_key):
#     """Any numeric param set to non-numeric raises ValueError."""
#     base = {
#         "target_name": "Star",
#         "total_observation_length_h": "1",
#         "exposure_NUV_s": "1",
#         "exposure_VIS_s": "1",
#         "exposure_IR_s": "1",
#     }
#     base[bad_key] = "not_a_number"
#     content = "\n".join(f"{k} = {v}" for k, v in base.items())
#     path = _write_params(tmp_path, content)
#     with pytest.raises(ValueError, match=f"{bad_key} must be a number"):
#         load_parameters(path)


# # --- Empty target_name ---
# def test_load_parameters_empty_target_name_raises(tmp_path):
#     content = """
#     target_name =
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     with pytest.raises(ValueError, match="target_name must not be empty"):
#         load_parameters(path)


# def test_load_parameters_target_name_only_quotes_raises(tmp_path):
#     content = """
#     target_name = "   "
#     total_observation_length_h = 1
#     exposure_NUV_s = 1
#     exposure_VIS_s = 1
#     exposure_IR_s = 1
#     """
#     path = _write_params(tmp_path, content)
#     with pytest.raises(ValueError, match="target_name must not be empty"):
#         load_parameters(path)



# def test_target_name_with_planet_letter_is_sanitized(tmp_path):
#     p = tmp_path / "parameters.txt"
#     p.write_text(
#         "target_name = HF 123 b\n"
#         "total_observation_length_h = 1\n"
#         "exposure_NUV_s = 1\n"
#         "exposure_VIS_s = 1\n"
#         "exposure_IR_s = 1\n"
#     )

#     params = load_parameters(p)
#     assert params["target_name"] == "HF 123 b"



# @pytest.mark.parametrize(
#     "target_name, expected",
#     [
#         ("HF 123", "HF 123"),
#         (" HF 123 ", "HF 123"),
#         ("'HF 123'", "HF 123"),
#         ('"HF 123"', "HF 123"),
#         ("' HF 123 '", "HF 123"),
#         ("HF 123B", "HF 123B"),
#     ],
# )
# def test_target_name_accepts_star_only_variants(tmp_path, target_name, expected):
#     path = _write_min_params_with_target(tmp_path, target_name)
#     params = load_parameters(path)
#     assert params["target_name"] == expected


# @pytest.mark.parametrize(
#     "target_name, expected",
#     [
#         ("HF 123 b", "HF 123 b"),
#         ("HF 123 B", "HF 123 B"),
#         ("HF 123 b ", "HF 123 b"),
#         ("'HF 123 b'", "HF 123 b"),
#         ('"HF 123 b"', "HF 123 b"),
#         ("' HF 123 b '", "HF 123 b"),

#         # --- star names with multiple internal spaces ---
#         ("HF   123", "HF   123"),          # 3 spaces inside
#         ("'HF   123'", "HF   123"),        # quotes + internal spaces
#         ('HF    123', "HF    123"),        # 4 spaces inside
#         ("'HF    123 '", "HF    123"),     # quotes + trailing space trimmed
#     ],
# )
# def test_target_name_sanitizes_but_does_not_reject(tmp_path, target_name, expected):
#     path = _write_min_params_with_target(tmp_path, target_name)
#     params = load_parameters(path)
#     assert params["target_name"] == expected
