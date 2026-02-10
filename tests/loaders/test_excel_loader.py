"""Tests for excel_loader: load_matching_excel_row_from_excel, map_to_planet_or_star_dictionary, _strip_hash_from_excel_column."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from loaders.excel_loader import (
    load_matching_excel_row_from_excel,
    load_excel_cfg,
    _strip_hash_from_excel_column,
)


def _write_excel(path: Path, headers: list, rows: list[list]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)

def test_map_to_planet_or_star_dictionary_logs_unknown_headers(tmp_path, caplog):
    from loaders.excel_loader import map_to_planet_or_star_dictionary

    # Excel row contains an unmapped header "weird_col"
    row = {
        "pl_name": "Planet X b",
        "st_teff": 5000,
        "weird_col": 123,   # unmapped
    }

    mapping = {
        "planet": {"orbital_period": "pl_orbper"},
        "star": {"effective_temperature": "st_teff"},
        "required_planetary_parameters": [],
        "required_stellar_parameters": [],
    }

    with caplog.at_level("WARNING"):
        planet_params, star_params = map_to_planet_or_star_dictionary(row, mapping, "Planet X")

    assert "Ignoring unmapped Excel columns" in " ".join(caplog.messages)
    assert planet_params == {}
    assert star_params["effective_temperature"] == 5000
    assert star_params["name"] == "Planet X"


def test_map_to_planet_or_star_dictionary_detects_missing_required_keys():
    from loaders.excel_loader import map_to_planet_or_star_dictionary

    row = {
        "pl_name": "Planet X b",
        "st_teff": 5000,
    }

    mapping = {
        "planet": {"orbital_period": "pl_orbper"},  # missing in row
        "star": {"effective_temperature": "st_teff"},
        "required_planetary_parameters": ["orbital_period"],
        "required_stellar_parameters": ["effective_temperature", "radius"],
    }

    planet_params, star_params = map_to_planet_or_star_dictionary(row, mapping, "Star X")

    # orbital_period missing
    assert "orbital_period" not in planet_params

    # radius missing
    assert "radius" not in star_params

    # star name still inserted
    assert star_params["name"] == "Star X"



def test_map_to_planet_or_star_dictionary_inserts_star_name():
    from loaders.excel_loader import map_to_planet_or_star_dictionary

    row = {
        "pl_name": "Planet X b",
        "st_teff": 5000,
    }

    mapping = {
        "planet": {},
        "star": {"effective_temperature": "st_teff"},
        "required_planetary_parameters": [],
        "required_stellar_parameters": [],
    }

    _, star_params = map_to_planet_or_star_dictionary(row, mapping, "Star X")

    assert star_params["name"] == "Star X"
    assert star_params["effective_temperature"] == 5000

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


# --- load_excel_cfg ---
def test_load_excel_cfg_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "excel_mapping.cfg"
    with pytest.raises(FileNotFoundError, match="Excel mapping file not found"):
        load_excel_cfg(missing_path)


# --- _strip_hash_from_excel_column ---
def test_strip_hash_from_excel_column_returns_none_for_none() -> None:
    assert _strip_hash_from_excel_column(None) is None


def test_strip_hash_from_excel_column_strips_whitespace() -> None:
    assert _strip_hash_from_excel_column("  pl_name  ") == "pl_name"


def test_strip_hash_from_excel_column_removes_leading_hash() -> None:
    assert _strip_hash_from_excel_column("#pl_name") == "pl_name"

def test_pl_name_does_not_match_on_prefix_only(tmp_path: Path) -> None:
    """
    Regression: previously we used startswith(), so a short prefix like
    'TIC 3938183' could match the wrong star ('TIC 393818343 b').
    We now require exact basename match.
    """
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[
            ["TIC 393818343 b", 5000],
            ["TIC 393818399 b", 5100],
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        load_matching_excel_row_from_excel(path, "TIC 3938183")

    assert "No target found" in str(exc_info.value)
    assert "TIC 3938183" in str(exc_info.value)


def test_pl_name_matches_star_basename_ignoring_planet_designator(tmp_path: Path) -> None:
    """
    Star basename should match even if Excel rows are planet rows (b/c/d...).
    """
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff", "pl_orbper"],
        rows=[
            ["TIC 393818343 b", 5000, 3.4],
            ["TIC 393818343 c", 5050, 7.8],
        ],
    )

    row_dict, target_name = load_matching_excel_row_from_excel(path, "TIC 393818343")

    assert target_name == "TIC 393818343"
    assert row_dict["pl_name"] in ("TIC 393818343 b", "TIC 393818343 c")

def test_excel_strips_planet_designator_and_returns_star_basename(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HF 123 b", 5000]],
    )

    row_dict, target_name = load_matching_excel_row_from_excel(path, "HF 123")

    assert row_dict["pl_name"] == "HF 123 b"
    assert target_name == "HF 123"

def test_excel_internal_spaces_preserved_in_basename(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HF   123    b", 5000]],
    )

    _, target_name = load_matching_excel_row_from_excel(path, "HF   123")
    assert target_name == "HF   123"

def test_excel_uppercase_planet_letter(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HF 123 B", 5000]],
    )

    _, target_name = load_matching_excel_row_from_excel(path, "HF 123")
    assert target_name == "HF 123"

def test_excel_trailing_whitespace_in_pl_name(tmp_path: Path) -> None:
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["HF 123 b   ", 5000]],
    )

    _, target_name = load_matching_excel_row_from_excel(path, "HF 123")
    assert target_name == "HF 123"

def test_excel_strips_planet_designator_for_lettered_stars(tmp_path: Path) -> None:
    """
    KELT-19 A b must match KELT-19 A.
    The Excel loader must strip the trailing planet letter and return the star basename.
    """
    path = tmp_path / "targets.xlsx"
    _write_excel(
        path,
        headers=["pl_name", "st_teff"],
        rows=[["KELT-19 A b", 5000]],
    )

    row_dict, target_name = load_matching_excel_row_from_excel(path, "KELT-19 A")

    assert row_dict["pl_name"] == "KELT-19 A b"
    assert target_name == "KELT-19 A"
