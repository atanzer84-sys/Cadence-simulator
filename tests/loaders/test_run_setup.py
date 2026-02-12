"""Tests for helpers in loaders.run_setup (Excel file selection and loading)."""

from pathlib import Path

import pytest
import sys
from loaders import run_setup


def test_setup_output_directory_creates_dir(monkeypatch, tmp_path):
    # Redirect "output" to a temp directory
    monkeypatch.chdir(tmp_path)

    output_dir, timestamp = run_setup.setup_output_directory()

    # Directory exists
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Timestamp looks correct
    assert len(timestamp) > 0
    assert timestamp in output_dir.name

def test_setup_output_directory_handles_collision(monkeypatch, tmp_path):
    # Freeze datetime.now() so both calls use the same timestamp
    class FixedDateTime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2025, 2, 5, 12, 0, 0, 0)

        @staticmethod
        def strftime(fmt):
            return FixedDateTime.now().strftime(fmt)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_setup, "datetime", FixedDateTime)

    # First call creates output/<timestamp>
    first_dir, ts = run_setup.setup_output_directory()
    assert first_dir.name == ts

    # Manually create the base directory again to force a collision
    base_dir = tmp_path / "output" / ts
    base_dir.mkdir(exist_ok=True)

    # Second call should now create <timestamp>_01
    second_dir, ts2 = run_setup.setup_output_directory()

    assert ts == ts2
    assert second_dir.name == f"{ts}_01"
    assert second_dir.exists()

def test_setup_output_directory_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    run_setup.setup_output_directory()
    captured = capsys.readouterr().out

    assert "Output directory created at:" in captured

def test_setup_logger_prints(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_setup.setup_logger(output_dir, timestamp)

    captured = capsys.readouterr().out
    # new: absolute path is printed
    assert "Log file created at:" in captured
    assert f"waltzer_simulator_{timestamp}.log" in captured

def test_setup_logger_creates_file(tmp_path):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_setup.setup_logger(output_dir, timestamp)

    log_file = output_dir / f"waltzer_simulator_{timestamp}.log"
    assert log_file.exists()


# --- _find_excel_file --------------------------------------------------------
def test_find_excel_file_no_excel(tmp_path: Path) -> None:
    """No *.xlsx files -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        run_setup._find_excel_file(tmp_path)

    assert "No Excel file found" in str(exc_info.value)

def test_find_excel_file_single_excel(tmp_path: Path) -> None:
    """Exactly one *.xlsx file -> that path is returned."""
    excel = tmp_path / "Targets_V10p1.xlsx"
    excel.write_bytes(b"")  # empty file is enough for this test

    found = run_setup._find_excel_file(tmp_path)

    assert found == excel

def test_find_excel_file_multiple_excels(tmp_path: Path) -> None:
    """More than one *.xlsx file -> ValueError with names listed."""
    (tmp_path / "A.xlsx").write_bytes(b"")
    (tmp_path / "B.xlsx").write_bytes(b"")

    with pytest.raises(ValueError) as exc_info:
        run_setup._find_excel_file(tmp_path)

    msg = str(exc_info.value)
    assert "Multiple Excel files found" in msg
    assert "A.xlsx" in msg
    assert "B.xlsx" in msg


# --- load_stellar_and_planetary_properties ---------------------------------------------------


def test_load_stellar_and_planetary_properties_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: (_ for _ in ()).throw(FileNotFoundError("no excel")))

    with pytest.raises(FileNotFoundError):
        run_setup.load_stellar_and_planetary_properties("Target", object())


def test_load_stellar_and_planetary_properties_raises_value_error(monkeypatch):
    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(run_setup, "load_matching_excel_row_from_excel", lambda _p, _t: (_ for _ in ()).throw(ValueError("bad excel")))

    with pytest.raises(ValueError):
        run_setup.load_stellar_and_planetary_properties("Target", object())


def test_get_user_parameter_path_too_many_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "a", "b"])

    with pytest.raises(SystemExit) as exc:
        run_setup.get_user_parameter_path()

    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "Usage:" in (out.out + out.err)

def test_get_user_parameter_path_default_file(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    (input_dir / "parameters.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])

    p = run_setup.get_user_parameter_path()

    assert p == Path("input") / "parameters.txt"
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "parameters.txt" in (captured.out + captured.err)

def test_get_user_parameter_path_custom_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "custom.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "custom.txt"])

    p = run_setup.get_user_parameter_path()

    assert p == Path("custom.txt")
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "custom.txt" in (captured.out + captured.err)

def test_get_user_parameter_path_missing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])  # default parameters.txt does not exist

    with pytest.raises(SystemExit) as exc:
        run_setup.get_user_parameter_path()

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

    monkeypatch.setattr(run_setup.ascii, "read", lambda *_args, **_kw: FakeTable())

    star_params = {"effective_temperature": 6801.0}
    with caplog.at_level("INFO"):
        out = run_setup.infer_mamajek_spectral_type(star_params, "dummy_path.txt")

    assert out["spectral_type"] == "F2V"
    assert "Loading Mamajek table" in " ".join(caplog.messages)
