"""Tests for waltzer_simulator CLI: argument handling, param file location, errors."""

import pytest
import waltzer_simulator
from configs import user_config

def _write_params(path, content: str) -> None:
    path.write_text(content.strip(), encoding="utf-8")

VALID_PARAMS_CONTENT = """
target_name = HD 202772 A
total_observation_length_h = 20.5
exposure_NUV_s = 3
exposure_VIS_s = 4.25
exposure_IR_s = 10
"""



def test_excel_error_exits(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "parameters.txt").write_text(VALID_PARAMS_CONTENT.strip(), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _: (_ for _ in ()).throw(ValueError("excel broken"))
    )

    with pytest.raises(SystemExit):
        waltzer_simulator.main()

    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Input error" in out
    assert "excel broken" in out

def test_excel_value_error_exits(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    params = input_dir / "parameters.txt"
    params.write_text(
        "target_name = HD 202772 A\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _: (_ for _ in ()).throw(ValueError("excel broken"))
    )

    with pytest.raises(SystemExit) as exc:
        waltzer_simulator.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Input error" in out
    assert "excel broken" in out

def test_excel_file_not_found_exits(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    params = input_dir / "parameters.txt"
    params.write_text(
        "target_name = HD 202772 A\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _: (_ for _ in ()).throw(ValueError("no excel"))
    )

    with pytest.raises(SystemExit) as exc:
        waltzer_simulator.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Input error" in out
    assert "no excel" in out



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



# --- One argument: param file path (different locations) ---
def test_one_argument_absolute_path_success(monkeypatch, tmp_path):
    # IMPORTANT: patch the symbol used by waltzer_simulator (it imports calculateFluxOnEarth directly)
    def fake_calculateFluxOnEarth(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(waltzer_simulator, "calculateFluxOnEarth", fake_calculateFluxOnEarth)

    param_file = tmp_path / "params.txt"
    _write_params(param_file, VALID_PARAMS_CONTENT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])

    waltzer_simulator.main()

    output_root = tmp_path / "output"
    assert output_root.exists()

    run_dirs = [p for p in output_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    log_files = list(run_dirs[0].glob("waltzer_simulator_*.log"))
    assert len(log_files) == 1




def test_one_argument_relative_path_success(monkeypatch, tmp_path, capsys):
    # IMPORTANT: patch the symbol used by waltzer_simulator (it imports calculateFluxOnEarth directly)
    def fake_calculateFluxOnEarth(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(waltzer_simulator, "calculateFluxOnEarth", fake_calculateFluxOnEarth)

    subdir = tmp_path / "param"
    subdir.mkdir()
    param_file = subdir / "Wasp 99.txt"
    _write_params(param_file, VALID_PARAMS_CONTENT)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "param/Wasp 99.txt"])

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
    # IMPORTANT: clear cached config so this test actually reads bad.txt
    monkeypatch.setattr(user_config, "_USER", None)

    param_file = tmp_path / "bad.txt"
    param_file.write_text(
        "target_name = Star\n"
        "total_observation_length_h = not_a_number\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])

    with pytest.raises(SystemExit) as exc_info:
        waltzer_simulator.main()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "Input error" in (out.out + out.err)


def test_main_calls_star_and_planet_constructors(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    params = input_dir / "parameters.txt"
    params.write_text(
        "target_name = HD 202772 A\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
       lambda _target: (
            {"name": "Planet"},   # planet_param
            {"name": "Star"},     # stellar_param
            ["name"],             # required_planetary_parameters
            ["name"],             # required_stellar_parameters
        ),
    )

    calls = {
        "star_from_params": None,
        "planet_from_params": None,
        "flux": None,
    }

    star_obj = object()
    planet_obj = object()

    def fake_star_from_params(stellar_param, required_keys):
        calls["star_from_params"] = (stellar_param, required_keys)
        return star_obj

    def fake_planet_from_params(planet_param, required_keys):
        calls["planet_from_params"] = (planet_param, required_keys)
        return planet_obj

    def fake_calculateFluxOnEarth(star, output_dir):
        calls["flux"] = (star, output_dir)
        return None

    monkeypatch.setattr(waltzer_simulator.Star, "from_params", staticmethod(fake_star_from_params))
    monkeypatch.setattr(waltzer_simulator.Planet, "from_params", staticmethod(fake_planet_from_params))
    monkeypatch.setattr(waltzer_simulator, "calculateFluxOnEarth", fake_calculateFluxOnEarth)

    waltzer_simulator.main()

    assert calls["star_from_params"] == ({"name": "Star"}, ["name"])
    assert calls["planet_from_params"] == ({"name": "Planet"}, ["name"])
    assert calls["flux"] is not None
    assert calls["flux"][0] is star_obj
