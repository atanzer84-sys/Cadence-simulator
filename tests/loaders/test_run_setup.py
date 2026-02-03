"""Tests for helpers in loaders.run_setup (Excel file selection and loading)."""

from pathlib import Path

import pytest

from loaders import run_setup


# --- _find_excel_file --------------------------------------------------------
def test_find_excel_file_no_excel(tmp_path: Path) -> None:
    """No *.xlsx files -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        run_setup._find_excel_file(base_dir=tmp_path)

    assert "No Excel file found" in str(exc_info.value)


def test_find_excel_file_single_excel(tmp_path: Path) -> None:
    """Exactly one *.xlsx file -> that path is returned."""
    excel = tmp_path / "Targets_V10p1.xlsx"
    excel.write_bytes(b"")  # empty file is enough for this test

    found = run_setup._find_excel_file(base_dir=tmp_path)

    assert found == excel


def test_find_excel_file_multiple_excels(tmp_path: Path) -> None:
    """More than one *.xlsx file -> ValueError with names listed."""
    (tmp_path / "A.xlsx").write_bytes(b"")
    (tmp_path / "B.xlsx").write_bytes(b"")

    with pytest.raises(ValueError) as exc_info:
        run_setup._find_excel_file(base_dir=tmp_path)

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
    monkeypatch.setattr(run_setup, "_find_excel_file", lambda base_dir=None: excel)

    called = {}

    def fake_loader(path, target):
        called["path"] = path
        called["target"] = target
        return ({"ok": True}, target)

    def fake_split(_dict, _target):
        return ({"stellar": True}, {"planetary": True})

    monkeypatch.setattr(run_setup, "load_excel_parameters", fake_loader)
    monkeypatch.setattr(run_setup, "separate_stellar_planetary_parameters", fake_split)
    monkeypatch.setattr(run_setup, "process_planetary_parameter_values", lambda d: d)
    monkeypatch.setattr(run_setup, "process_stellar_parameter_values", lambda d: d)

    target_name = "HD 202772 A"
    with caplog.at_level("INFO"):
        result = run_setup.load_Excel_properties(target_name)

    # load_Excel_properties returns (planet_param, stellar_param) — planetary first, stellar second
    assert result == ({"planetary": True}, {"stellar": True})
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

    monkeypatch.setattr(run_setup, "_find_excel_file", lambda base_dir=None: excel)

    seen = {}

    def fake_loader(path, target):
        seen["path"] = path
        seen["target"] = target
        return ({"ok": True}, target)

    def fake_split(_dict, _target):
        return ({"stellar": True}, {"planetary": True})

    monkeypatch.setattr(run_setup, "load_excel_parameters", fake_loader)
    monkeypatch.setattr(run_setup, "separate_stellar_planetary_parameters", fake_split)
    monkeypatch.setattr(run_setup, "process_planetary_parameter_values", lambda d: d)
    monkeypatch.setattr(run_setup, "process_stellar_parameter_values", lambda d: d)

    target_name = ""
    with caplog.at_level("INFO"):
        result = run_setup.load_Excel_properties(target_name)

    # load_Excel_properties returns (planet_param, stellar_param) — planetary first, stellar second
    assert result == ({"planetary": True}, {"stellar": True})
    assert seen["path"] == excel
    assert seen["target"] == target_name

    log_text = " ".join(caplog.messages)
    assert "Using Excel file" in log_text
    assert "Targets.xlsx" in log_text
