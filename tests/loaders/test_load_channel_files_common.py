"""
Direct tests for common low-level calibration file loaders.
"""

from pathlib import Path

import numpy as np
import pytest

from loaders.load_channel_files_common import (
    find_first_numeric_row_index,
    load_background_file,
    load_background_from_global_cfg,
    load_effective_area_file,
    load_zod_dist_file,
    load_zod_spectrum_file,
    parse_spread_header_wavelengths,
    read_text_lines_with_fallback,
)

_REPO_ROOT_COMMON = "loaders.load_channel_files_common.get_repo_root"
_GLOBAL_CFG_COMMON = "loaders.load_channel_files_common.get_global_config"


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_COMMON, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


def _write(path: Path, text: str) -> None:
    """Write raw text to file (for edge-case test content)."""
    path.write_text(text, encoding="utf-8")


def _write_ea_file(path: Path, pixel_scale: float = 0.01, rows: int = 3) -> None:
    """Write a minimal effective area file (wavelength, effective_area columns)."""
    lines = [f"# Pixel scale: {pixel_scale}", "Wavelength  EffectiveArea"]
    for i in range(rows):
        wl = 1000.0 + i * 100
        ea = 0.1 + i * 0.1
        lines.append(f"{wl}  {ea}")
    path.write_text("\n".join(lines), encoding="utf-8")


# Tests: load_effective_area_file
# Behavior: loads wavelength, effective area, and pixel scale from valid file
def test_load_effective_area_file_success(data_dir):
    _write_ea_file(data_dir / "ea.txt", pixel_scale=0.05, rows=4)

    wavelength, effective_area, pixel_scale, spectral_dispersion_A_per_pixel = load_effective_area_file("ea.txt")

    assert len(wavelength) == 4
    assert np.allclose(wavelength, [1000.0, 1100.0, 1200.0, 1300.0])
    assert np.allclose(effective_area, [0.1, 0.2, 0.3, 0.4])
    assert pixel_scale == pytest.approx(0.05)
    assert spectral_dispersion_A_per_pixel == pytest.approx(100.0)


# Tests: load_effective_area_file
# Behavior: missing file raises ValueError
def test_load_effective_area_file_missing_file_raises(data_dir):
    with pytest.raises(ValueError):
        load_effective_area_file("nonexistent.txt")


# Tests: load_effective_area_file
# Behavior: missing pixel scale header raises ValueError
def test_load_effective_area_file_missing_pixel_scale_raises(data_dir):
    _write(data_dir / "ea.txt", "1000  0.1\n1100  0.2\n")

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_effective_area_file
# Behavior: missing numeric rows raises ValueError
def test_load_effective_area_file_no_numeric_data_raises(data_dir):
    _write(data_dir / "ea.txt", "# Pixel scale: 0.01\n# only comments\nWavelength  EA\n")

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_effective_area_file
# Behavior: invalid pixel scale value raises ValueError
def test_load_effective_area_file_invalid_pixel_scale_raises(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: not_a_number\nWavelength EffectiveArea\n1000  0.1\n1100  0.2\n",
    )

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_effective_area_file
# Behavior: one-column table raises ValueError
def test_load_effective_area_file_one_column_table_raises(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.417684\nWavelength\n2397.0000\n2397.4177\n2397.8354\n",
    )

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_effective_area_file
# Behavior: single-row two-column table raises ValueError
def test_load_effective_area_file_one_row_two_columns_raises(data_dir):
    _write(data_dir / "ea.txt", "# Pixel scale: 0.5\nWavelength EffectiveArea\n1500  0.42\n")

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_effective_area_file
# Behavior: extra header lines after pixel scale are tolerated
def test_load_effective_area_file_header_lines_after_pixel_scale_ok(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.02\n# another comment\n# yet another\n"
        "Wavelength foo bar EffectiveArea\n1000  7  8  0.11\n1100  9 10  0.22\n",
    )

    wl, ea, pixel_scale, spectral_dispersion_A_per_pixel = load_effective_area_file("ea.txt")

    assert pixel_scale == pytest.approx(0.02)
    assert spectral_dispersion_A_per_pixel == pytest.approx(100.0)
    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.11, 0.22])


# Tests: load_effective_area_file
# Behavior: extra columns use first and last columns
def test_load_effective_area_file_extra_columns_first_and_last_used(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength c2 c3 c4 EffectiveArea\n1000  1  2  3  0.50\n1100  4  5  6  0.60\n",
    )

    wl, ea, _, _ = load_effective_area_file("ea.txt")

    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.50, 0.60])


# Tests: load_effective_area_file
# Behavior: leading and trailing whitespace is tolerated
def test_load_effective_area_file_leading_trailing_whitespace_ok(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength EffectiveArea\n   1000\t   0.10   \n\t1100\t0.20\t\n",
    )

    wl, ea, _, _ = load_effective_area_file("ea.txt")

    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.10, 0.20])


# Tests: load_effective_area_file
# Behavior: blank lines inside numeric block are tolerated
def test_load_effective_area_file_blank_lines_inside_numeric_block_ok(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength EffectiveArea\n1000  0.10\n\n1100  0.20\n   \n1200  0.30\n",
    )

    wl, ea, _, _ = load_effective_area_file("ea.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(ea, [0.10, 0.20, 0.30])


# Tests: load_effective_area_file
# Behavior: malformed numeric row raises ValueError
def test_load_effective_area_file_malformed_numeric_row_raises(data_dir):
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength foo EffectiveArea\n1000  1  0.10\n1100  2  BAD\n1200  3  0.30\n",
    )

    with pytest.raises(ValueError):
        load_effective_area_file("ea.txt")


# Tests: load_background_file
# Behavior: empty or blank filename returns none values
def test_load_background_file_empty_filename_returns_none_none():
    assert load_background_file("") == (None, None)
    assert load_background_file("   ") == (None, None)


# Tests: load_background_file
# Behavior: loads wavelength and flux from valid file
def test_load_background_file_success(data_dir):
    _write(data_dir / "bg.txt", "1000  0.1\n1100  0.2\n1200  0.3\n")

    wl, flux = load_background_file("bg.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(flux, [0.1, 0.2, 0.3])


# Tests: load_background_file
# Behavior: missing file raises ValueError
def test_load_background_file_missing_file_raises(data_dir):
    with pytest.raises(ValueError):
        load_background_file("missing_bg.txt")


# Tests: load_background_file
# Behavior: one-column table raises ValueError
def test_load_background_file_one_column_table_raises(data_dir):
    _write(data_dir / "bg.txt", "1000\n1100\n")

    with pytest.raises(ValueError):
        load_background_file("bg.txt")


# Tests: load_background_file
# Behavior: malformed numeric row raises ValueError
def test_load_background_file_malformed_numeric_row_raises(data_dir):
    _write(
        data_dir / "bg.txt",
        "1000  0.1\n1100  BAD\n1200  0.3\n",
    )

    with pytest.raises(ValueError):
        load_background_file("bg.txt")


# Tests: load_background_from_global_cfg
# Behavior: empty background type returns empty background payload
def test_load_background_from_global_cfg_empty_type_returns_empty_payload(make_global_config, monkeypatch):
    cfg = make_global_config(background_type="", sky_pixel_area_arcsec2=None)

    monkeypatch.setattr(_GLOBAL_CFG_COMMON, lambda: cfg)

    background = load_background_from_global_cfg()

    assert background["background_type"] is None
    assert background["background_wavelength"] is None
    assert background["background_flux"] is None
    assert background["sky_pixel_area_arcsec2"] is None
    assert background["zod_dist"] is None
    assert background["zod_spectrum_wavelength"] is None
    assert background["zod_spectrum_flux"] is None


# Tests: load_background_from_global_cfg
# Behavior: default background loads spectrum and sky pixel area
def test_load_background_from_global_cfg_default_loads_background_file(make_global_config, monkeypatch):
    cfg = make_global_config(
        background_type="default",
        background_file="background.txt",
        sky_pixel_area_arcsec2=2.5,
    )

    monkeypatch.setattr(_GLOBAL_CFG_COMMON, lambda: cfg)
    monkeypatch.setattr(
        "loaders.load_channel_files_common.load_background_file",
        lambda filename: (np.array([1000.0, 1100.0]), np.array([0.1, 0.2])),
    )

    background = load_background_from_global_cfg()

    assert background["background_type"] == "default"
    assert np.allclose(background["background_wavelength"], [1000.0, 1100.0])
    assert np.allclose(background["background_flux"], [0.1, 0.2])
    assert background["sky_pixel_area_arcsec2"] == 2.5
    assert background["zod_dist"] is None
    assert background["zod_spectrum_wavelength"] is None
    assert background["zod_spectrum_flux"] is None


# Tests: load_background_from_global_cfg
# Behavior: calc background loads zodiacal inputs
def test_load_background_from_global_cfg_calc_loads_zodiacal_inputs(make_global_config, monkeypatch):
    cfg = make_global_config(
        background_type="calc",
        zod_dist_file="zod_dist.txt",
        zod_spectrum_file="zod_spec.txt",
        sky_pixel_area_arcsec2=3.0,
    )

    monkeypatch.setattr(_GLOBAL_CFG_COMMON, lambda: cfg)
    monkeypatch.setattr(
        "loaders.load_channel_files_common.load_zod_dist_file",
        lambda filename: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    monkeypatch.setattr(
        "loaders.load_channel_files_common.load_zod_spectrum_file",
        lambda filename: (np.array([1200.0, 1300.0]), np.array([0.3, 0.4])),
    )

    background = load_background_from_global_cfg()

    assert background["background_type"] == "calc"
    assert background["background_wavelength"] is None
    assert background["background_flux"] is None
    assert background["sky_pixel_area_arcsec2"] == 3.0
    assert np.allclose(background["zod_dist"], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(background["zod_spectrum_wavelength"], [1200.0, 1300.0])
    assert np.allclose(background["zod_spectrum_flux"], [0.3, 0.4])


# Tests: load_background_from_global_cfg
# Behavior: invalid background type raises ValueError
def test_load_background_from_global_cfg_invalid_type_raises(make_global_config, monkeypatch):
    cfg = make_global_config(background_type="weird")

    monkeypatch.setattr(_GLOBAL_CFG_COMMON, lambda: cfg)

    with pytest.raises(ValueError):
        load_background_from_global_cfg()


# Tests: load_zod_dist_file
# Behavior: empty or blank filename returns none
def test_load_zod_dist_file_no_file_provided_returns_none():
    assert load_zod_dist_file("") is None
    assert load_zod_dist_file("   ") is None


# Tests: load_zod_dist_file
# Behavior: loads and transposes valid table
def test_load_zod_dist_file_success(data_dir):
    _write(data_dir / "zod_dist.txt", "1.0  2.0  3.0\n4.0  5.0  6.0\n")

    data = load_zod_dist_file("zod_dist.txt")

    assert data.ndim == 2
    assert data.shape == (3, 2)
    assert np.allclose(data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


# Tests: load_zod_dist_file
# Behavior: missing file raises ValueError
def test_load_zod_dist_file_missing_file_raises(data_dir):
    with pytest.raises(ValueError):
        load_zod_dist_file("missing_zod.txt")


# Tests: load_zod_dist_file
# Behavior: one-column table raises ValueError
def test_load_zod_dist_file_one_column_raises(data_dir):
    _write(data_dir / "zod.txt", "1.0\n2.0\n3.0\n")

    with pytest.raises(ValueError):
        load_zod_dist_file("zod.txt")


# Tests: load_zod_dist_file
# Behavior: empty file raises ValueError
def test_load_zod_dist_file_empty_file_raises(data_dir):
    _write(data_dir / "zod.txt", "")

    with pytest.raises(ValueError):
        load_zod_dist_file("zod.txt")


# Tests: load_zod_dist_file
# Behavior: malformed row raises ValueError
def test_load_zod_dist_file_malformed_row_raises(data_dir):
    _write(data_dir / "zod.txt", "1.0  2.0\n3.0  BAD\n5.0  6.0\n")

    with pytest.raises(ValueError):
        load_zod_dist_file("zod.txt")


# Tests: load_zod_spectrum_file
# Behavior: empty or blank filename returns none values
def test_load_zod_spectrum_file_no_file_provided_returns_none_none():
    assert load_zod_spectrum_file("") == (None, None)
    assert load_zod_spectrum_file("   ") == (None, None)


# Tests: load_zod_spectrum_file
# Behavior: loads wavelength and spectrum from valid file
def test_load_zod_spectrum_file_success(data_dir):
    _write(data_dir / "zod_spec.txt", "1000  0.5\n1100  0.6\n1200  0.7\n")

    wl, spec = load_zod_spectrum_file("zod_spec.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(spec, [0.5, 0.6, 0.7])


# Tests: load_zod_spectrum_file
# Behavior: missing file raises ValueError
def test_load_zod_spectrum_file_missing_file_raises(data_dir):
    with pytest.raises(ValueError):
        load_zod_spectrum_file("missing_spec.txt")


# Tests: load_zod_spectrum_file
# Behavior: one-column table raises ValueError
def test_load_zod_spectrum_file_one_column_raises(data_dir):
    _write(data_dir / "zod_spec.txt", "1000\n1100\n1200\n")

    with pytest.raises(ValueError):
        load_zod_spectrum_file("zod_spec.txt")


# Tests: load_zod_spectrum_file
# Behavior: empty file raises ValueError
def test_load_zod_spectrum_file_empty_file_raises(data_dir):
    _write(data_dir / "zod_spec.txt", "")

    with pytest.raises(ValueError):
        load_zod_spectrum_file("zod_spec.txt")


# Tests: load_zod_spectrum_file
# Behavior: malformed row raises ValueError
def test_load_zod_spectrum_file_malformed_row_raises(data_dir):
    _write(data_dir / "zod_spec.txt", "1000  0.5\n1100  BAD\n1200  0.7\n")

    with pytest.raises(ValueError):
        load_zod_spectrum_file("zod_spec.txt")


# Tests: find_first_numeric_row_index
# Behavior: returns first numeric row index
def test_find_first_numeric_row_index_success():
    lines = [
        "# Pixel scale: 0.01",
        "Wavelength EffectiveArea",
        "",
        "1000  0.1",
        "1100  0.2",
    ]

    idx = find_first_numeric_row_index(lines, Path("ea.txt"))

    assert idx == 3


# Tests: find_first_numeric_row_index
# Behavior: missing numeric rows raises ValueError
def test_find_first_numeric_row_index_no_numeric_rows_raises():
    lines = [
        "# Pixel scale: 0.01",
        "Wavelength EffectiveArea",
        "",
        "# comment",
    ]

    with pytest.raises(ValueError):
        find_first_numeric_row_index(lines, Path("ea.txt"))


# Tests: parse_spread_header_wavelengths
# Behavior: parses wavelengths from pixels header
def test_parse_spread_header_wavelengths_success():
    lines = [
        "# comment",
        "",
        "pixels 1000 1100 1200",
        "1.0 2.0 3.0",
    ]

    wavelength_header = parse_spread_header_wavelengths(lines, Path("spread.txt"), "VIS")

    assert np.allclose(wavelength_header, [1000.0, 1100.0, 1200.0])


# Tests: parse_spread_header_wavelengths
# Behavior: missing pixels header raises ValueError
def test_parse_spread_header_wavelengths_missing_header_raises():
    lines = [
        "# comment",
        "",
        "Wavelength 1000 1100 1200",
    ]

    with pytest.raises(ValueError):
        parse_spread_header_wavelengths(lines, Path("spread.txt"), "VIS")


# Tests: read_text_lines_with_fallback
# Behavior: falls back to later encoding
def test_read_text_lines_with_fallback_utf16_success(tmp_path):
    path = tmp_path / "utf16.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-16")

    lines = read_text_lines_with_fallback(path, ("utf-8", "utf-16"), "test")

    assert lines == ["alpha", "beta"]


# Tests: load_effective_area_file
# Behavior: falls back to UTF-16 and parses numeric table from decoded lines
def test_load_effective_area_file_encoding_fallback_utf16_success(data_dir):
    path = data_dir / "ea_utf16.txt"
    text = (
        "# Pixel scale: 0.05\n"
        "Wavelength EffectiveArea\n"
        "1000 0.1\n"
        "1100 0.2\n"
        "1200 0.3\n"
    )
    path.write_text(text, encoding="utf-16")

    wl, ea, pixel_scale, spectral_dispersion_A_per_pixel = load_effective_area_file("ea_utf16.txt")

    assert pixel_scale == pytest.approx(0.05)
    assert spectral_dispersion_A_per_pixel == pytest.approx(100.0)
    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(ea, [0.1, 0.2, 0.3])


# Tests: load_background_file
# Behavior: empty file raises ValueError
def test_load_background_file_empty_file_raises(data_dir):
    _write(data_dir / "bg.txt", "")

    with pytest.raises(ValueError):
        load_background_file("bg.txt")


# Tests: load_background_file
# Behavior: extra columns use first two columns
def test_load_background_file_extra_columns_first_two_used(data_dir):
    _write(data_dir / "bg.txt", "1000  0.1  9.0\n1100  0.2  8.0\n1200  0.3  7.0\n")

    wl, flux = load_background_file("bg.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(flux, [0.1, 0.2, 0.3])


# Tests: load_background_file
# Behavior: blank lines and whitespace are tolerated
def test_load_background_file_blank_lines_and_whitespace_ok(data_dir):
    _write(data_dir / "bg.txt", " 1000\t0.1 \n\n 1100  0.2 \n   \n1200\t 0.3\n")

    wl, flux = load_background_file("bg.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(flux, [0.1, 0.2, 0.3])


# Tests: load_zod_dist_file
# Behavior: single numeric row raises ValueError
def test_load_zod_dist_file_single_row_raises(data_dir):
    _write(data_dir / "zod.txt", "1.0  2.0  3.0\n")

    with pytest.raises(ValueError):
        load_zod_dist_file("zod.txt")


# Tests: load_zod_dist_file
# Behavior: blank lines and whitespace are tolerated
def test_load_zod_dist_file_blank_lines_and_whitespace_ok(data_dir):
    _write(data_dir / "zod.txt", "1.0  2.0  3.0\n\n 4.0  5.0  6.0 \n")

    data = load_zod_dist_file("zod.txt")

    assert data.ndim == 2
    assert data.shape == (3, 2)
    assert np.allclose(data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


# Tests: load_zod_spectrum_file
# Behavior: extra columns use first two columns
def test_load_zod_spectrum_file_extra_columns_first_two_used(data_dir):
    _write(data_dir / "zod_spec.txt", "1000  0.5  9.0\n1100  0.6  8.0\n1200  0.7  7.0\n")

    wl, spec = load_zod_spectrum_file("zod_spec.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(spec, [0.5, 0.6, 0.7])


# Tests: load_zod_spectrum_file
# Behavior: blank lines and whitespace are tolerated
def test_load_zod_spectrum_file_blank_lines_and_whitespace_ok(data_dir):
    _write(data_dir / "zod_spec.txt", "1000  0.5\n\n 1100  0.6 \n   \n1200\t0.7\n")

    wl, spec = load_zod_spectrum_file("zod_spec.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(spec, [0.5, 0.6, 0.7])

# Tests: load_zod_spectrum_file
# Behavior: single-row two-column table raises ValueError
def test_load_zod_spectrum_file_single_row_two_columns_raises(data_dir):
    _write(data_dir / "zod_spec.txt", "1000  0.5\n")

    with pytest.raises(ValueError):
        load_zod_spectrum_file("zod_spec.txt")

# Tests: load_background_file
# Behavior: single-row two-column table raises ValueError
def test_load_background_file_single_row_two_columns_raises(data_dir):
    _write(data_dir / "bg.txt", "1000  0.1\n")

    with pytest.raises(ValueError):
        load_background_file("bg.txt")

# Tests: parse_spread_header_wavelengths
# Behavior: malformed wavelength token raises ValueError
def test_parse_spread_header_wavelengths_malformed_wavelength_token_raises():
    lines = [
        "# comment",
        "",
        "pixels 1000 BAD 1200",
    ]

    with pytest.raises(ValueError):
        parse_spread_header_wavelengths(lines, Path("spread.txt"), "VIS")

# Tests: parse_spread_header_wavelengths
# Behavior: pixels header without wavelength columns raises ValueError
def test_parse_spread_header_wavelengths_no_wavelength_columns_raises():
    lines = [
        "# comment",
        "",
        "pixels",
    ]

    with pytest.raises(ValueError):
        parse_spread_header_wavelengths(lines, Path("spread.txt"), "VIS")

# Tests: read_text_lines_with_fallback
# Behavior: utf-8-sig encoded file is read successfully
def test_read_text_lines_with_fallback_utf8_sig_success(tmp_path):
    path = tmp_path / "utf8sig.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8-sig")

    lines = read_text_lines_with_fallback(path, ("utf-8", "utf-8-sig"), "test")

    assert lines[0].lstrip("\ufeff") == "alpha"
    assert lines[1] == "beta"

# Tests: read_text_lines_with_fallback
# Behavior: all encoding attempts failing raises ValueError
def test_read_text_lines_with_fallback_all_encodings_fail_raises(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\xff\xfe\xfa\xfb")

    with pytest.raises(ValueError):
        read_text_lines_with_fallback(path, ("utf-8", "ascii"), "test")