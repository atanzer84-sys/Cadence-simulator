"""
Direct tests for low-level calibration file loaders:
- load_effective_area_file
- load_spread_profile_file
- load_background_file
- load_zod_dist_file
- load_zod_spectrum_file
"""

from pathlib import Path

import numpy as np
import pytest

from loaders.load_channel_files import (
    load_effective_area_file,
    load_spread_profile_file,
    load_background_file,
    load_zod_dist_file,
    load_zod_spectrum_file,
)

_REPO_ROOT = "loaders.load_channel_files.get_repo_root"


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


def _write_spread_file(path: Path, wavelengths: list[float], num_rows: int = 3) -> None:
    """Write a minimal spread profile file (pixels header + dy + weight columns)."""
    wl_str = "  ".join(str(w) for w in wavelengths)
    lines = ["# comment", f"pixels  {wl_str}"]
    for i in range(num_rows):
        dy = float(i)
        weights = "  ".join("0.5" for _ in wavelengths)
        lines.append(f"{dy}  {weights}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# load_effective_area_file: direct tests
# ----------------------------------------------------------------------


def test_load_effective_area_file_success(monkeypatch, tmp_path):
    """load_effective_area_file loads wavelength, effective_area, pixel_scale from real file."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ea_path = data_dir / "ea.txt"
    _write_ea_file(ea_path, pixel_scale=0.05, rows=4)

    wavelength, effective_area, pixel_scale = load_effective_area_file("ea.txt")

    assert len(wavelength) == 4
    assert np.allclose(wavelength, [1000.0, 1100.0, 1200.0, 1300.0])
    assert np.allclose(effective_area, [0.1, 0.2, 0.3, 0.4])
    assert pixel_scale == pytest.approx(0.05)


def test_load_effective_area_file_missing_file_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError when file does not exist."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Effective area file not found"):
        load_effective_area_file("nonexistent.txt")


def test_load_effective_area_file_missing_pixel_scale_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError when '# Pixel scale:' header is missing."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ea_path = data_dir / "ea.txt"
    ea_path.write_text("1000  0.1\n1100  0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required header line"):
        load_effective_area_file("ea.txt")


def test_load_effective_area_file_no_numeric_data_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError when no numeric data rows exist."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ea_path = data_dir / "ea.txt"
    ea_path.write_text("# Pixel scale: 0.01\n# only comments\nWavelength  EA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not find any numeric data"):
        load_effective_area_file("ea.txt")


def test_load_effective_area_file_invalid_pixel_scale_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError when pixel scale value is non-numeric."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: not_a_number\nWavelength EffectiveArea\n1000  0.1\n1100  0.2\n",
    )

    with pytest.raises(ValueError, match="Invalid pixel scale value"):
        load_effective_area_file("ea.txt")


def test_load_effective_area_file_one_column_table_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError for one-column table (ndim==1 guard)."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.417684\nWavelength\n2397.0000\n2397.4177\n2397.8354\n",
    )

    with pytest.raises(ValueError, match="Invalid effective area table structure"):
        load_effective_area_file("ea.txt")


def test_load_effective_area_file_one_row_two_columns_raises(monkeypatch, tmp_path):
    """load_effective_area_file raises ValueError for single numeric row (np.loadtxt returns 1D)."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "ea.txt", "# Pixel scale: 0.5\nWavelength EffectiveArea\n1500  0.42\n")

    with pytest.raises(ValueError, match="Invalid effective area table structure"):
        load_effective_area_file("ea.txt")


def test_load_effective_area_file_header_lines_after_pixel_scale_ok(monkeypatch, tmp_path):
    """Extra comment lines after pixel scale header do not break parsing."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.02\n# another comment\n# yet another\n"
        "Wavelength foo bar EffectiveArea\n1000  7  8  0.11\n1100  9 10  0.22\n",
    )

    wl, ea, pixel_scale = load_effective_area_file("ea.txt")
    assert pixel_scale == pytest.approx(0.02)
    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.11, 0.22])


def test_load_effective_area_file_extra_columns_first_and_last_used(monkeypatch, tmp_path):
    """With extra columns, first column is wavelength and last column is effective area."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength c2 c3 c4 EffectiveArea\n1000  1  2  3  0.50\n1100  4  5  6  0.60\n",
    )

    wl, ea, _ = load_effective_area_file("ea.txt")
    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.50, 0.60])


def test_load_effective_area_file_leading_trailing_whitespace_ok(monkeypatch, tmp_path):
    """Whitespace and tabs around numeric values do not break parsing."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength EffectiveArea\n   1000\t   0.10   \n\t1100\t0.20\t\n",
    )

    wl, ea, _ = load_effective_area_file("ea.txt")
    assert np.allclose(wl, [1000.0, 1100.0])
    assert np.allclose(ea, [0.10, 0.20])


def test_load_effective_area_file_blank_lines_inside_numeric_block_ok(monkeypatch, tmp_path):
    """Blank lines between numeric rows are tolerated."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength EffectiveArea\n1000  0.10\n\n1100  0.20\n   \n1200  0.30\n",
    )

    wl, ea, _ = load_effective_area_file("ea.txt")
    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(ea, [0.10, 0.20, 0.30])


def test_load_effective_area_file_malformed_numeric_row_raises(monkeypatch, tmp_path):
    """Malformed numeric row causes parse failure rather than silent skip."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "ea.txt",
        "# Pixel scale: 0.01\nWavelength foo EffectiveArea\n1000  1  0.10\n1100  2  BAD\n1200  3  0.30\n",
    )

    with pytest.raises(ValueError, match="Failed to parse numeric data"):
        load_effective_area_file("ea.txt")


# ----------------------------------------------------------------------
# load_spread_profile_file: direct tests
# ----------------------------------------------------------------------


def test_load_spread_profile_file_success(monkeypatch, tmp_path):
    """load_spread_profile_file loads positions, weights, wavelengths from real file."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spread_path = data_dir / "spread.txt"
    _write_spread_file(spread_path, wavelengths=[1000.0, 1100.0], num_rows=3)

    positions, weights, wavelengths = load_spread_profile_file("spread.txt", "NUV")

    assert np.allclose(positions, [0.0, 1.0, 2.0])
    assert weights.shape == (3, 2)
    assert np.allclose(wavelengths, [1000.0, 1100.0])


def test_load_spread_profile_file_missing_file_raises(monkeypatch, tmp_path):
    """load_spread_profile_file raises ValueError when file does not exist (non-empty filename)."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Spread profile file not found"):
        load_spread_profile_file("nonexistent_spread.txt", "NUV")


def test_load_spread_profile_file_missing_pixels_header_raises(monkeypatch, tmp_path):
    """load_spread_profile_file raises ValueError when 'pixels' header line is absent."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "spread.txt", "# comment\n0  0.1 0.2\n1  0.3 0.4\n")

    with pytest.raises(ValueError, match="No 'pixels"):
        load_spread_profile_file("spread.txt", "NUV")


def test_load_spread_profile_file_header_count_mismatch_raises(monkeypatch, tmp_path):
    """load_spread_profile_file raises ValueError when header wavelength count != weight columns."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "spread.txt", "pixels 1000 1100 1200\n0  0.1 0.2\n1  0.3 0.4\n")

    with pytest.raises(ValueError, match="wavelength count does not match weight columns"):
        load_spread_profile_file("spread.txt", "NUV")


def test_load_spread_profile_file_leading_trailing_whitespace_ok(monkeypatch, tmp_path):
    """Whitespace and blank lines do not break spread parsing; output dtypes are float."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "spread.txt",
        "   pixels   1000   1100   \n\n  0    0.10   0.20   \n  1    0.30   0.40   \n\n",
    )

    pos, w, wl = load_spread_profile_file("spread.txt", "NUV")
    assert pos.dtype == float
    assert w.dtype == float
    assert wl.dtype == float
    assert pos.shape == (2,)
    assert w.shape == (2, 2)
    assert wl.shape == (2,)
    assert wl[0] == pytest.approx(1000.0)
    assert wl[1] == pytest.approx(1100.0)
    assert w[1, 1] == pytest.approx(0.40)


# ----------------------------------------------------------------------
# load_background_file: direct tests
# ----------------------------------------------------------------------


def test_load_background_file_empty_filename_returns_none_none():
    """Empty or blank background_filename returns (None, None) without touching the filesystem."""
    assert load_background_file("") == (None, None)
    assert load_background_file("   ") == (None, None)


def test_load_background_file_success(monkeypatch, tmp_path):
    """load_background_file loads wavelength and flux from a simple two-column table."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bg_path = data_dir / "bg.txt"
    _write(bg_path, "1000  0.1\n1100  0.2\n1200  0.3\n")

    wl, flux = load_background_file("bg.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(flux, [0.1, 0.2, 0.3])


def test_load_background_file_missing_file_raises(monkeypatch, tmp_path):
    """load_background_file raises ValueError when the file does not exist."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Background file not found"):
        load_background_file("missing_bg.txt")


def test_load_background_file_one_column_table_raises(monkeypatch, tmp_path):
    """One-column numeric table is rejected as invalid background structure."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "bg.txt", "1000\n1100\n")

    with pytest.raises(ValueError, match="Invalid background table structure"):
        load_background_file("bg.txt")


def test_load_background_file_malformed_numeric_row_raises(monkeypatch, tmp_path):
    """Malformed numeric row causes parse failure rather than silent skip."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "bg.txt",
        "1000  0.1\n1100  BAD\n1200  0.3\n",
    )

    with pytest.raises(ValueError, match="Failed to parse numeric data"):
        load_background_file("bg.txt")


# ----------------------------------------------------------------------
# load_zod_dist_file: direct tests
# ----------------------------------------------------------------------


def test_load_zod_dist_file_empty_filename_returns_none():
    """Empty or blank filename returns None without touching the filesystem."""
    assert load_zod_dist_file("") is None
    assert load_zod_dist_file("   ") is None


def test_load_zod_dist_file_success(monkeypatch, tmp_path):
    """load_zod_dist_file loads a 2D numeric table from file."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod_dist.txt", "1.0  2.0  3.0\n4.0  5.0  6.0\n")

    data = load_zod_dist_file("zod_dist.txt")

    assert data.ndim == 2
    assert data.shape == (2, 3)
    assert np.allclose(data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_load_zod_dist_file_missing_file_raises(monkeypatch, tmp_path):
    """load_zod_dist_file raises ValueError when the file does not exist."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Zodiacal distribution file not found"):
        load_zod_dist_file("missing_zod.txt")


def test_load_zod_dist_file_one_column_raises(monkeypatch, tmp_path):
    """Single-column data yields 1D array and is rejected as invalid table."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod.txt", "1.0\n2.0\n3.0\n")

    with pytest.raises(ValueError, match="Invalid zodiacal distribution table"):
        load_zod_dist_file("zod.txt")


def test_load_zod_dist_file_empty_file_raises(monkeypatch, tmp_path):
    """Empty file (no numeric rows) raises ValueError."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod.txt", "")

    with pytest.raises(ValueError):
        load_zod_dist_file("zod.txt")


def test_load_zod_dist_file_malformed_row_raises(monkeypatch, tmp_path):
    """Malformed numeric row causes parse failure."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod.txt", "1.0  2.0\n3.0  BAD\n5.0  6.0\n")

    with pytest.raises(ValueError, match="Failed to parse"):
        load_zod_dist_file("zod.txt")


# ----------------------------------------------------------------------
# load_zod_spectrum_file: direct tests
# ----------------------------------------------------------------------


def test_load_zod_spectrum_file_empty_filename_returns_none_none():
    """Empty or blank filename returns (None, None) without touching the filesystem."""
    assert load_zod_spectrum_file("") == (None, None)
    assert load_zod_spectrum_file("   ") == (None, None)


def test_load_zod_spectrum_file_success(monkeypatch, tmp_path):
    """load_zod_spectrum_file loads wavelength and spectrum columns from file."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod_spec.txt", "1000  0.5\n1100  0.6\n1200  0.7\n")

    wl, spec = load_zod_spectrum_file("zod_spec.txt")

    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(spec, [0.5, 0.6, 0.7])


def test_load_zod_spectrum_file_missing_file_raises(monkeypatch, tmp_path):
    """load_zod_spectrum_file raises ValueError when the file does not exist."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Zodiacal spectrum file not found"):
        load_zod_spectrum_file("missing_spec.txt")


def test_load_zod_spectrum_file_one_column_raises(monkeypatch, tmp_path):
    """One-column table is rejected as invalid spectrum structure."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod_spec.txt", "1000\n1100\n1200\n")

    with pytest.raises(ValueError, match="Invalid zodiacal spectrum table"):
        load_zod_spectrum_file("zod_spec.txt")


def test_load_zod_spectrum_file_empty_file_raises(monkeypatch, tmp_path):
    """Empty file (no numeric rows) raises ValueError."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod_spec.txt", "")

    with pytest.raises(ValueError):
        load_zod_spectrum_file("zod_spec.txt")


def test_load_zod_spectrum_file_malformed_row_raises(monkeypatch, tmp_path):
    """Malformed numeric row causes parse failure."""
    monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "zod_spec.txt", "1000  0.5\n1100  BAD\n1200  0.7\n")

    with pytest.raises(ValueError, match="Failed to parse"):
        load_zod_spectrum_file("zod_spec.txt")

