"""Tests for excel_loader: load_matching_excel_row_from_excel, map_to_planet_or_star_dictionary, _normalize_name."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from loaders.excel_loader import (
    load_matching_excel_row_from_excel,
    load_excel_cfg,
    _normalize_name,
)


def _write_excel(path: Path, headers: list, rows: list[list]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)


# --- load_matching_excel_row_from_excel: pl_name not found ---
def test_pl_name_not_found_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["Other Star b", 5000]],
    )
    with pytest.raises(ValueError) as exc_info:
        load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert "No target found" in str(exc_info.value)
    assert "HD 202772 A" in str(exc_info.value)


# --- load_matching_excel_row_from_excel: no pl_name column ---
def test_excel_no_pl_name_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["planet", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    with pytest.raises(ValueError) as exc_info:
        load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert "no 'pl_name' column" in str(exc_info.value)


# --- load_matching_excel_row_from_excel: empty pl_name in row ---
def test_excel_empty_pl_name_in_row_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[[None, 5000]],  # empty pl_name in first data row
    )
    with pytest.raises(ValueError) as exc_info:
        load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert "No target found" in str(exc_info.value)


# --- load_matching_excel_row_from_excel: one found ---
def test_pl_name_one_found_returns_row_and_target_name(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff", "pl_orbper"],
        rows=[["HD 202772 A b", 5000, 3.4]],
    )
    row_dict, target_name = load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000
    assert row_dict.get("pl_orbper") == 3.4
    assert target_name == "HD 202772 A"


# --- load_matching_excel_row_from_excel: two found (first match wins) ---
def test_pl_name_two_found_first_wins(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[
            ["HD 202772 A b", 5000],
            ["HD 202772 A c", 5100],
        ],
    )
    row_dict, _ = load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000


# --- load_matching_excel_row_from_excel: case insensitive ---
def test_pl_name_match_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    row_dict, _ = load_matching_excel_row_from_excel(path, "hd 202772 a")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"


def test_pl_name_match_target_uppercase_row_mixed(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    row_dict, _ = load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"




# --- load_matching_excel_row_from_excel: header normalization (# prefix) ---
def test_excel_header_with_hash_normalized(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["#pl_name", "st_teff"])  # # is stripped from header
    ws.append(["HD 202772 A b", 5000])
    wb.save(path)
    row_dict, _ = load_matching_excel_row_from_excel(path, "HD 202772 A")
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000


# --- load_matching_excel_row_from_excel: returns stripped target_name ---
def test_load_excel_returns_stripped_target_name(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    _, target_name = load_matching_excel_row_from_excel(path, "  HD 202772 A  ")
    assert target_name == "HD 202772 A"


# --- load_excel_cfg ---
def test_load_excel_cfg_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "excel_mapping.cfg"
    with pytest.raises(FileNotFoundError, match="Excel mapping file not found"):
        load_excel_cfg(missing_path)


# --- _normalize_name ---
def test_normalize_name_returns_none_for_none() -> None:
    assert _normalize_name(None) is None


def test_normalize_name_strips_whitespace() -> None:
    assert _normalize_name("  pl_name  ") == "pl_name"


def test_normalize_name_removes_leading_hash() -> None:
    assert _normalize_name("#pl_name") == "pl_name"
