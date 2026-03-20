"""Tests for loaders.run_waltzer_context."""

import sys
from pathlib import Path

import pytest

from loaders import run_waltzer_context


def test_setup_output_directory_creates_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    output_dir, timestamp_str, _ = run_waltzer_context.setup_output_directory()

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

    first_dir, ts1, _ = run_waltzer_context.setup_output_directory()
    second_dir, ts2, _ = run_waltzer_context.setup_output_directory()

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
    assert "Log file created at:" in captured
    assert f"waltzer_simulator_{timestamp}.log" in captured


def test_setup_logger_creates_file(tmp_path):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_waltzer_context.setup_logger(output_dir, timestamp)

    log_file = output_dir / f"waltzer_simulator_{timestamp}.log"
    assert log_file.exists()


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
    """Custom parameter file must be under repo root (path validation)."""
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "custom.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "input/custom.txt"])

    p = run_waltzer_context.get_user_parameter_path()

    assert p.resolve() == (tmp_path / "input" / "custom.txt").resolve()
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "custom.txt" in (captured.out + captured.err)


def test_get_user_parameter_path_missing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])

    with pytest.raises(SystemExit) as exc:
        run_waltzer_context.get_user_parameter_path()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "parameter file not found" in (captured.out + captured.err).lower()


def test_get_user_parameter_path_one_argument_absolute_path_success(monkeypatch, tmp_path):
    """Absolute path to param file must be under repo root (path validation)."""
    (tmp_path / "input").mkdir(exist_ok=True)
    param_file = tmp_path / "input" / "params.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])
    monkeypatch.chdir(tmp_path)

    got = run_waltzer_context.get_user_parameter_path()

    assert got.resolve() == param_file.resolve()


def test_get_user_parameter_path_one_argument_relative_path_success(monkeypatch, tmp_path):
    """Relative path to param file must resolve under repo root (path validation)."""
    subdir = tmp_path / "param"
    subdir.mkdir()
    param_file = subdir / "Wasp 99.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "param/Wasp 99.txt"])

    got = run_waltzer_context.get_user_parameter_path()

    assert got.resolve() == param_file.resolve()


def test_get_user_parameter_path_no_argument_file_not_found_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1


def test_get_user_parameter_path_rejects_path_traversal(monkeypatch, tmp_path):
    """Path outside repo root raises ValueError (path traversal protection)."""
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "../../../etc/passwd"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert "Path traversal" in str(exc_info.value) or "outside" in str(exc_info.value).lower()


def test_one_argument_file_not_found_exits(monkeypatch, tmp_path, capsys):
    """Path under repo root but file missing raises SystemExit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "input/nonexistent_params.txt"])

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "not found" in text.lower()


def test_invalid_params_raises_value_error(monkeypatch, tmp_path):
    import configs.user_config as user_config

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
        run_waltzer_context.load_global_and_user_config()

    assert "total_observation_length_h" in str(exc_info.value)
