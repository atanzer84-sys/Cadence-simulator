"""Tests for waltzer_simulator CLI: argument handling, param file location, errors."""

import pytest

import waltzer_simulator


def _write_params(path, content: str) -> None:
    path.write_text(content.strip(), encoding="utf-8")


VALID_PARAMS_CONTENT = """
target_name = HD 202772 A
total_observation_length_h = 20.5
exposure_NUV_s = 3
exposure_VIS_s = 4.25
exposure_IR_s = 10
"""


# --- Too many arguments ---
def test_too_many_arguments_exits_with_usage(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # keep output/ and log out of project root
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "a.txt", "b.txt"])
    with pytest.raises(SystemExit) as exc_info:
        waltzer_simulator.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "Usage:" in out.out or "Usage:" in out.err
    assert "parameters_file" in (out.out + out.err)


# --- No argument: default parameters.txt ---
def test_no_argument_file_not_found_exits(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])
    monkeypatch.chdir(tmp_path)  # empty dir, no parameters.txt
    with pytest.raises(SystemExit) as exc_info:
        waltzer_simulator.main()
    assert exc_info.value.code == 1


def test_no_argument_uses_default_file_success(monkeypatch, tmp_path, capsys):
    (tmp_path / "parameters.txt").write_text(VALID_PARAMS_CONTENT.strip(), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])
    monkeypatch.chdir(tmp_path)
    # Avoid real Excel lookup (repo root has no .xlsx in test env); main() can then complete.
    monkeypatch.setattr(
        waltzer_simulator,
        "load_excel_properties",
        lambda _: (
            {"name": "Planet"},
            {
                "name": "Star",
                "effective_temperature": 1.0,
                "radius": 1.0,
                "mass": 1.0,
                "surface_gravity": 1.0,
                "right_ascension": 1.0,
                "declination": 1.0,
                "distance": 1.0,
            },
            ["name"],
            [
                "name",
                "effective_temperature",
                "radius",
                "mass",
                "surface_gravity",
                "right_ascension",
                "declination",
                "distance",
            ],
        ),
    )
    waltzer_simulator.main()
    out = capsys.readouterr()
    assert "target_name" in out.out
    assert "HD 202772 A" in out.out


# --- One argument: param file path (different locations) ---
def test_one_argument_absolute_path_success(monkeypatch, tmp_path, capsys):
    param_file = tmp_path / "params.txt"
    _write_params(param_file, VALID_PARAMS_CONTENT)
    monkeypatch.chdir(tmp_path)  # keep output/ and log out of project root
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])
    waltzer_simulator.main()
    out = capsys.readouterr()
    assert "HD 202772 A" in out.out
    assert "20.5" in out.out


def test_one_argument_relative_path_success(monkeypatch, tmp_path, capsys):
    subdir = tmp_path / "param"
    subdir.mkdir()
    param_file = subdir / "Wasp 99.txt"
    _write_params(param_file, VALID_PARAMS_CONTENT)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "param/Wasp 99.txt"])
    monkeypatch.chdir(tmp_path)
    waltzer_simulator.main()
    out = capsys.readouterr()
    assert "HD 202772 A" in out.out


def test_one_argument_file_not_found_exits(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # keep output/ and log out of project root
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "/nonexistent/params.txt"])
    with pytest.raises(SystemExit) as exc_info:
        waltzer_simulator.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "not found" in (out.out + out.err)
    assert "nonexistent" in (out.out + out.err) or "parameter file" in (out.out + out.err)


# --- Invalid param file (ValueError) ---
def test_one_argument_invalid_params_exits(monkeypatch, tmp_path, capsys):
    param_file = tmp_path / "bad.txt"
    param_file.write_text(
        "target_name = Star\ntotal_observation_length_h = not_a_number\n"
        "exposure_NUV_s = 1\nexposure_VIS_s = 1\nexposure_IR_s = 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)  # keep output/ and log out of project root
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])
    with pytest.raises(SystemExit) as exc_info:
        waltzer_simulator.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "Input error" in (out.out + out.err)
