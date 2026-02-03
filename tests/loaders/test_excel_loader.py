"""Tests for excel_loader: load_excel_parameters, separate_stellar_planetary_parameters, _normalize_name."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from loaders.excel_loader import (
    load_excel_parameters,
    separate_stellar_planetary_parameters,
    _normalize_name,
)


def _write_excel(path: Path, headers: list, rows: list[list]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)


# --- load_excel_parameters: pl_name not found ---
def test_pl_name_not_found_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["Other Star b", 5000]],
    )
    with pytest.raises(ValueError) as exc_info:
        load_excel_parameters(path, "HD 202772 A")
    assert "No target found" in str(exc_info.value)
    assert "HD 202772 A" in str(exc_info.value)


# --- load_excel_parameters: no pl_name column ---
def test_excel_no_pl_name_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["planet", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    with pytest.raises(ValueError) as exc_info:
        load_excel_parameters(path, "HD 202772 A")
    assert "no 'pl_name' column" in str(exc_info.value)


# --- load_excel_parameters: empty pl_name in row ---
def test_excel_empty_pl_name_in_row_raises(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[[None, 5000]],  # empty pl_name in first data row
    )
    with pytest.raises(ValueError) as exc_info:
        load_excel_parameters(path, "HD 202772 A")
    assert "Empty pl_name" in str(exc_info.value)


# --- load_excel_parameters: one found ---
def test_pl_name_one_found_returns_row_and_target_name(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff", "pl_orbper"],
        rows=[["HD 202772 A b", 5000, 3.4]],
    )
    row_dict, target_name = load_excel_parameters(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000
    assert row_dict.get("pl_orbper") == 3.4
    assert target_name == "HD 202772 A"


# --- load_excel_parameters: two found (first match wins) ---
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
    row_dict, _ = load_excel_parameters(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000


# --- load_excel_parameters: case insensitive ---
def test_pl_name_match_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    row_dict, _ = load_excel_parameters(path, "hd 202772 a")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"


def test_pl_name_match_target_uppercase_row_mixed(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    row_dict, _ = load_excel_parameters(path, "HD 202772 A")
    assert row_dict is not None
    assert row_dict.get("pl_name") == "HD 202772 A b"


def test_load_excel_parameters_skips_none_rows(monkeypatch) -> None:
    class DummyCell:
        def __init__(self, value):
            self.value = value

    class DummyWorksheet:
        def __init__(self):
            self._headers = [DummyCell("pl_name"), DummyCell("st_teff")]
            self._rows = [None, ("HD 202772 A b", 5000)]

        def __getitem__(self, item):
            if item == 1:
                return self._headers
            raise KeyError(item)

        def iter_rows(self, min_row=2, values_only=True):
            for row in self._rows:
                yield row

    class DummyWorkbook:
        active = DummyWorksheet()

    monkeypatch.setattr(
        "loaders.excel_loader.load_workbook",
        lambda _path, data_only=True: DummyWorkbook(),
    )

    row_dict, target_name = load_excel_parameters("dummy.xlsx", "HD 202772 A")
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000
    assert target_name == "HD 202772 A"


# --- load_excel_parameters: header normalization (# prefix) ---
def test_excel_header_with_hash_normalized(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["#pl_name", "st_teff"])  # # is stripped from header
    ws.append(["HD 202772 A b", 5000])
    wb.save(path)
    row_dict, _ = load_excel_parameters(path, "HD 202772 A")
    assert row_dict.get("pl_name") == "HD 202772 A b"
    assert row_dict.get("st_teff") == 5000


# --- load_excel_parameters: returns stripped target_name ---
def test_load_excel_returns_stripped_target_name(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HD 202772 A b", 5000]],
    )
    _, target_name = load_excel_parameters(path, "  HD 202772 A  ")
    assert target_name == "HD 202772 A"


# --- separate_stellar_planetary_parameters ---
def test_separate_stellar_planetary_pl_prefix_goes_to_planetary() -> None:
    d = {"pl_name": "b", "pl_orbper": 3.4, "st_teff": 5000}
    stellar, planetary = separate_stellar_planetary_parameters(d, "HD 202772 A")
    assert planetary["pl_name"] == "b"
    assert planetary["pl_orbper"] == 3.4
    assert "pl_name" not in stellar
    assert stellar["st_teff"] == 5000


def test_separate_stellar_planetary_st_and_others_go_to_stellar() -> None:
    d = {"st_teff": 5000, "st_rad": 1.1, "discoverymethod": "Transit"}
    stellar, planetary = separate_stellar_planetary_parameters(d, "Star")
    assert stellar["st_teff"] == 5000
    assert stellar["st_rad"] == 1.1
    assert planetary["discoverymethod"] == "Transit"


def test_separate_stellar_planetary_planet_keys_go_to_planetary() -> None:
    d = {"discoverymethod": "Transit", "scale_height_km": 100, "st_teff": 5000}
    stellar, planetary = separate_stellar_planetary_parameters(d, "Star")
    assert planetary["discoverymethod"] == "Transit"
    assert planetary["scale_height_km"] == 100
    assert stellar["st_teff"] == 5000


def test_separate_stellar_planetary_sets_st_name() -> None:
    d = {"st_teff": 5000}
    stellar, planetary = separate_stellar_planetary_parameters(d, "HD 202772 A")
    assert stellar["st_name"] == "HD 202772 A"
    assert "st_name" not in planetary


# --- _normalize_name ---
def test_normalize_name_returns_none_for_none() -> None:
    assert _normalize_name(None) is None


def test_normalize_name_strips_whitespace() -> None:
    assert _normalize_name("  pl_name  ") == "pl_name"


def test_normalize_name_removes_leading_hash() -> None:
    assert _normalize_name("#pl_name") == "pl_name"
