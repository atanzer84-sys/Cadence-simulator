"""Tests for helpers in loaders.run_waltzer_context (Excel file selection and loading)."""

from pathlib import Path

import pytest
import sys
from loaders import run_waltzer_context
import configs.global_config as gc
from configs.global_config import GlobalConfig
from loaders.load_stellar_and_planetary_properties import apply_log_r_fallback
from configs import user as user_config
import logging

def test_setup_output_directory_creates_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    output_dir, timestamp_str = run_waltzer_context.setup_output_directory()

    assert output_dir.exists()
    assert output_dir.is_dir()

    assert isinstance(timestamp_str, str)
    assert len(timestamp_str) > 0
    assert timestamp_str in output_dir.name


def test_setup_output_directory_handles_collision(monkeypatch, tmp_path):
    class FixedDateTime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2025, 2, 5, 12, 0, 0, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "datetime", FixedDateTime)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    first_dir, ts1 = run_waltzer_context.setup_output_directory()
    second_dir, ts2 = run_waltzer_context.setup_output_directory()

    assert first_dir.exists()
    assert second_dir.exists()

    assert ts1 == ts2
    assert first_dir != second_dir


def test_setup_output_directory_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    run_waltzer_context.setup_output_directory()
    captured = capsys.readouterr().out

    assert "Output directory created at:" in captured

def test_setup_logger_prints(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_waltzer_context.setup_logger(output_dir, timestamp)

    captured = capsys.readouterr().out
    # new: absolute path is printed
    assert "Log file created at:" in captured
    assert f"waltzer_simulator_{timestamp}.log" in captured

def setup_logger(output_dir, timestamp_str):
    filename = f"waltzer_simulator_{timestamp_str}.log"
    log_filename = output_dir / filename
    print(f"Log file created at: {log_filename.resolve()}")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s")
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)

    logging.info("Logger initialized")



def test_setup_logger_creates_file(tmp_path):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_waltzer_context.setup_logger(output_dir, timestamp)

    log_file = output_dir / f"waltzer_simulator_{timestamp}.log"
    assert log_file.exists()


# --- _find_excel_file --------------------------------------------------------
def test_find_excel_file_no_excel(tmp_path: Path) -> None:
    """No *.xlsx files -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        run_waltzer_context._find_excel_file(tmp_path)

    assert "No Excel file found" in str(exc_info.value)

def test_find_excel_file_single_excel(tmp_path: Path) -> None:
    """Exactly one *.xlsx file -> that path is returned."""
    excel = tmp_path / "Targets_V10p1.xlsx"
    excel.write_bytes(b"")  # empty file is enough for this test

    found = run_waltzer_context._find_excel_file(tmp_path)

    assert found == excel

def test_find_excel_file_multiple_excels(tmp_path: Path) -> None:
    """More than one *.xlsx file -> ValueError with names listed."""
    (tmp_path / "A.xlsx").write_bytes(b"")
    (tmp_path / "B.xlsx").write_bytes(b"")

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context._find_excel_file(tmp_path)

    msg = str(exc_info.value)
    assert "Multiple Excel files found" in msg
    assert "A.xlsx" in msg
    assert "B.xlsx" in msg


# --- load_stellar_and_planetary_properties ---------------------------------------------------


def test_load_stellar_and_planetary_properties_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_waltzer_context, "_find_excel_file", lambda repo_root: (_ for _ in ()).throw(FileNotFoundError("no excel")))

    with pytest.raises(FileNotFoundError):
        run_waltzer_context.load_stellar_and_planetary_properties("Target")


def test_load_stellar_and_planetary_properties_raises_value_error(monkeypatch):
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_waltzer_context, "_find_excel_file", lambda repo_root: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(run_waltzer_context, "load_matching_excel_row_from_excel", lambda _p, _t: (_ for _ in ()).throw(ValueError("bad excel")))

    with pytest.raises(ValueError):
        run_waltzer_context.load_stellar_and_planetary_properties("Target")

# --- Too many arguments ---
def test_too_many_arguments_exits_with_usage(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "a.txt", "b.txt"])

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Usage:" in text
    assert "parameters_file" in text

def test_get_user_parameter_path_default_file(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    (input_dir / "parameters.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    monkeypatch.setattr(sys, "argv", ["prog"])

    p = run_waltzer_context.get_user_parameter_path()

    assert p.resolve() == (tmp_path / "input" / "parameters.txt").resolve()
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "parameters.txt" in (captured.out + captured.err)

def test_get_user_parameter_path_custom_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "custom.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "custom.txt"])

    p = run_waltzer_context.get_user_parameter_path()

    assert p == Path("custom.txt")
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "custom.txt" in (captured.out + captured.err)

def test_get_user_parameter_path_missing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    
    monkeypatch.setattr(sys, "argv", ["prog"])  # default parameters.txt does not exist

    with pytest.raises(SystemExit) as exc:
        run_waltzer_context.get_user_parameter_path()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "parameter file not found" in (captured.out + captured.err).lower()

def test_infer_mamajek_spectral_type_sets_spectral_type(monkeypatch, caplog):
    # fake table with two rows
    class FakeTable:
        def __len__(self): return 2
        def __getitem__(self, key):
            if key == "col1": return ["F1V", "F2V"]
            if key == "col2": return [6900.0, 6800.0]
            raise KeyError(key)

    monkeypatch.setattr(run_waltzer_context.ascii, "read", lambda *_args, **_kw: FakeTable())

    star_params = {"effective_temperature": 6801.0}
    with caplog.at_level("INFO"):
        out = run_waltzer_context.infer_mamajek_spectral_type(star_params, "dummy_path.txt")

    assert out["spectral_type"] == "F2V"
    assert "Loading Mamajek table" in " ".join(caplog.messages)


@pytest.fixture
def global_cfg_log_r(monkeypatch):
    cfg = GlobalConfig(
        line_core_emission=False,
        interstellar_absorption=False,
        mg2_col=None,
        mg1_col=None,
        fe2_col=None,
        sigmaMg22=0.257,
        sigmaMg21=0.288,
        enable_log_r_fallback=True,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
        test_mode=True,
        produce_Plots=False,
        n_bias_and_darkframes = 0,
        write_dark_and_bias_png = False,
        n_science_frames_per_channel=1,
        write_science_frames_png=0)
    monkeypatch.setattr(gc, "_GLOBAL", cfg, raising=False)
    return cfg

@pytest.fixture
def global_cfg_log_r_disabled(monkeypatch):
    cfg = GlobalConfig(
        line_core_emission=False,
        interstellar_absorption=False,
        mg2_col=None,
        mg1_col=None,
        fe2_col=None,
        sigmaMg22=0.257,
        sigmaMg21=0.288,
        enable_log_r_fallback=False,
        log_r_teff_threshold=5500.0,
        log_r_hot_value=-4.2,
        log_r_cool_value=-4.8,
        test_mode=True,
        produce_Plots=False,
        n_bias_and_darkframes = 0,
        write_dark_and_bias_png = False,
        n_science_frames_per_channel=1,
        write_science_frames_png=0
    )
    monkeypatch.setattr(gc, "_GLOBAL", cfg, raising=False)
    return cfg

def test_apply_log_r_fallback_disabled_does_not_modify_dict(global_cfg_log_r_disabled):
    star_params = {"effective_temperature": 6000}
    out = apply_log_r_fallback(star_params)

    assert out is star_params
    assert "log_r" not in star_params


def test_apply_log_r_fallback_does_not_override_existing_log_r(global_cfg_log_r):
    star_params = {"effective_temperature": 6000, "log_r": -9.9}

    out = apply_log_r_fallback(star_params)

    assert out is star_params
    assert star_params["log_r"] == -9.9


def test_apply_log_r_fallback_sets_hot_or_cool_value_based_on_threshold(monkeypatch, global_cfg_log_r):
    # hot branch
    star_hot = {"effective_temperature": "6000"}
    out_hot = apply_log_r_fallback(star_hot)
    assert out_hot is star_hot
    assert star_hot["log_r"] == -4.2

    # cool branch
    star_cool = {"effective_temperature": "5000"}
    out_cool = apply_log_r_fallback(star_cool)
    assert out_cool is star_cool
    assert star_cool["log_r"] == -4.8




def test_get_user_parameter_path_one_argument_absolute_path_success(monkeypatch, tmp_path):
    param_file = tmp_path / "params.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])
    monkeypatch.chdir(tmp_path)

    got = run_waltzer_context.get_user_parameter_path()

    assert got == param_file


def test_get_user_parameter_path_one_argument_relative_path_success(monkeypatch, tmp_path):
    subdir = tmp_path / "param"
    subdir.mkdir()
    param_file = subdir / "Wasp 99.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "param/Wasp 99.txt"])

    got = run_waltzer_context.get_user_parameter_path()

    assert got == Path("param/Wasp 99.txt")
    assert got.resolve() == param_file.resolve()



def test_get_user_parameter_path_no_argument_file_not_found_exits(monkeypatch, tmp_path):
    # Default path is repo_root/input/parameters.txt
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])
    monkeypatch.chdir(tmp_path)  # doesn't matter, default is repo_root-based

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    
def test_one_argument_file_not_found_exits(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "/nonexistent/params.txt"])

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "not found" in text.lower()
    assert "nonexistent" in text or "parameter" in text.lower()


def test_invalid_params_raises_value_error(monkeypatch, tmp_path):
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
    monkeypatch.setattr(run_waltzer_context, "get_user_parameter_path", lambda: Path(param_file))

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.load_cfg_and_user_config()

    assert "Invalid float for key 'total_observation_length_h'" in str(exc_info.value)


def test_infer_mamajek_spectral_type_missing_teff_raises(monkeypatch):
    class FakeTable:
        def __len__(self): return 1
        def __getitem__(self, key):
            if key == "col1": return ["F1V"]
            if key == "col2": return [6900.0]
            raise KeyError(key)

    monkeypatch.setattr(run_waltzer_context.ascii, "read", lambda *_args, **_kw: FakeTable())

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.infer_mamajek_spectral_type({}, "dummy_path.txt")

    assert "effective_temperature" in str(exc_info.value)


def test_infer_mamajek_spectral_type_invalid_teff_raises(monkeypatch):
    class FakeTable:
        def __len__(self): return 1
        def __getitem__(self, key):
            if key == "col1": return ["F1V"]
            if key == "col2": return [6900.0]
            raise KeyError(key)

    monkeypatch.setattr(run_waltzer_context.ascii, "read", lambda *_args, **_kw: FakeTable())

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.infer_mamajek_spectral_type({"effective_temperature": "nope"}, "dummy_path.txt")

    assert "invalid effective_temperature" in str(exc_info.value)


def test_apply_log_r_fallback_missing_teff_does_not_modify_dict(global_cfg_log_r):
    star_params = {}
    out = apply_log_r_fallback(star_params)

    assert out is star_params
    assert "log_r" not in star_params


def test_apply_log_r_fallback_invalid_teff_does_not_modify_dict(global_cfg_log_r):
    star_params = {"effective_temperature": "nope"}
    out = apply_log_r_fallback(star_params)

    assert out is star_params
    assert "log_r" not in star_params


def test_find_excel_file_ignores_excel_lock_files(tmp_path: Path) -> None:
    (tmp_path / "~$Targets_V10p1.xlsx").write_bytes(b"")
    real = tmp_path / "Targets_V10p1.xlsx"
    real.write_bytes(b"")

    found = run_waltzer_context._find_excel_file(tmp_path)

    assert found == real


def test_merge_gaia_into_star_params_excel_wins_and_fills_blanks(caplog):
    star_params = {
        "teff": 5000,          # existing non-empty -> keep
        "log_g": "",           # empty string -> replace
        "radius": None,        # None -> replace
    }
    gaia_star_params = {
        "teff": 9999,
        "log_g": 4.5,
        "radius": 0.9,
    }

    out = run_waltzer_context.merge_gaia_into_star_params(star_params, gaia_star_params)

    assert out is star_params
    assert star_params["teff"] == 5000
    assert star_params["log_g"] == 4.5
    assert star_params["radius"] == 0.9


def test_merge_gaia_into_star_params_none_gaia_returns_original():
    star_params = {"teff": 5000}

    out = run_waltzer_context.merge_gaia_into_star_params(star_params, None)

    assert out is star_params
    assert star_params == {"teff": 5000}
