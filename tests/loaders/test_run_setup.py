"""Tests for helpers in loaders.run_setup (Excel file selection and loading)."""

from pathlib import Path

import pytest
import sys
from loaders import run_setup

import logging



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
    assert f"Log file created at: waltzer_simulator_{timestamp}.log" in captured

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


# --- load_Excel_properties ---------------------------------------------------


def test_load_excel_properties_with_target_name(monkeypatch, tmp_path, caplog):
    """Happy path: Excel file found and loader called with target."""
    excel = tmp_path / "Targets.xlsx"
    excel.write_bytes(b"")

    # Use our tmp directory instead of the real repo root.
    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: excel)

    called = {}

    def fake_loader(path, target):
        called["path"] = path
        called["target"] = target
        return ({"ok": True}, target)

    monkeypatch.setattr(run_setup, "load_matching_excel_row_from_excel", fake_loader)
    monkeypatch.setattr(
        run_setup,
        "load_excel_cfg",
        lambda _path: {
            "mapping": True,
            "required_star_parameters": [],
            "required_planet_parameters": [],
        },
    )
    monkeypatch.setattr(run_setup, "get_missing_properties", lambda _p, _req: [])
    monkeypatch.setattr(run_setup, "lookup_star_gaia", lambda _s: {})
    monkeypatch.setattr(run_setup, "merge_gaia_into_star_params", lambda s, _g: s)
    monkeypatch.setattr(run_setup, "clean_and_cast_parameters", lambda params, _cls: params)
    monkeypatch.setattr(
        run_setup,
        "map_to_planet_or_star_dictionary",
        lambda _row, _mapping, _target: ({"planetary": True}, {"stellar": True}),
    )

    target_name = "HD 202772 A"
    with caplog.at_level("INFO"):
        result = run_setup.load_excel_properties(target_name)

    # load_Excel_properties returns params plus required keys
    assert result == ({"planetary": True}, {"stellar": True}, [], [])
    assert called["path"] == excel
    assert called["target"] == target_name

    # Check that we logged which Excel file and target were used.
    log_text = " ".join(caplog.messages)
    assert "Using Excel file" in log_text
    assert "Targets.xlsx" in log_text
    assert target_name in log_text


def test_load_excel_properties_with_empty_target_name(monkeypatch, tmp_path, caplog):
    """Even an empty target name is passed through and logged."""
    excel = tmp_path / "Targets.xlsx"
    excel.write_bytes(b"")

    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: excel)

    seen = {}

    def fake_loader(path, target):
        seen["path"] = path
        seen["target"] = target
        return ({"ok": True}, target)

    monkeypatch.setattr(run_setup, "load_matching_excel_row_from_excel", fake_loader)
    monkeypatch.setattr(
        run_setup,
        "load_excel_cfg",
        lambda _path: {
            "mapping": True,
            "required_star_parameters": [],
            "required_planet_parameters": [],
        },
    )
    monkeypatch.setattr(run_setup, "get_missing_properties", lambda _p, _req: [])
    monkeypatch.setattr(run_setup, "lookup_star_gaia", lambda _s: {})
    monkeypatch.setattr(run_setup, "merge_gaia_into_star_params", lambda s, _g: s)
    monkeypatch.setattr(run_setup, "clean_and_cast_parameters", lambda params, _cls: params)
    monkeypatch.setattr(
        run_setup,
        "map_to_planet_or_star_dictionary",
        lambda _row, _mapping, _target: ({"planetary": True}, {"stellar": True}),
    )

    target_name = ""
    with caplog.at_level("INFO"):
        result = run_setup.load_excel_properties(target_name)

    # load_Excel_properties returns params plus required keys
    assert result == ({"planetary": True}, {"stellar": True}, [], [])
    assert seen["path"] == excel
    assert seen["target"] == target_name

    log_text = " ".join(caplog.messages)
    assert "Using Excel file" in log_text
    assert "Targets.xlsx" in log_text


def test_load_excel_properties_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: (_ for _ in ()).throw(FileNotFoundError("no excel")))

    with pytest.raises(FileNotFoundError):
        run_setup.load_excel_properties("Target")

def test_load_excel_properties_raises_value_error(monkeypatch):
    monkeypatch.setattr(run_setup, "get_repo_root", lambda base_dir=None: Path("/tmp"))
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda repo_root: Path("/tmp/Targets.xlsx"))
    monkeypatch.setattr(run_setup, "load_matching_excel_row_from_excel", lambda _p, _t: (_ for _ in ()).throw(ValueError("bad excel")))

    with pytest.raises(ValueError):
        run_setup.load_excel_properties("Target")

def test_load_user_parameters_default_file(monkeypatch, tmp_path):
    # Create default parameters.txt
    param_file = tmp_path / "parameters.txt"
    param_file.write_text("target_name = HD 202772 A", encoding="utf-8")

    # Simulate no arguments
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.chdir(tmp_path)

    # Fake loader so we don't depend on real parsing
    monkeypatch.setattr(run_setup, "load_parameters", lambda path: {"ok": True})

    result = run_setup.load_user_parameters()
    assert result == {"ok": True}
