from pathlib import Path
import numpy as np
import pytest

from loaders.load_channel_files_spectroscopy import load_spread_profile_file_spectroscopy

_REPO_ROOT_SPEC = "loaders.load_channel_files_spectroscopy.get_repo_root"


def _write(path: Path, text: str) -> None:
    """Write raw text to file (for edge-case test content)."""
    path.write_text(text, encoding="utf-8")


def _write_spread_file(path: Path, wavelengths: list[float], num_rows: int = 3) -> None:
    """Write a minimal spread profile file (pixels header + dy + weight columns)."""
    wl_str = "  ".join(str(w) for w in wavelengths)
    lines = ["# comment", f"pixels  {wl_str}"]
    for i in range(num_rows):
        dy = float(i)
        weights = "  ".join("0.5" for _ in wavelengths)
        lines.append(f"{dy}  {weights}")
    path.write_text("\n".join(lines), encoding="utf-8")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: loads positions, weights, and wavelengths from valid spread file
def test_load_spread_profile_file_spectroscopy_success(monkeypatch, tmp_path):
    """load_spread_profile_file_spectroscopy loads positions, weights, wavelengths from real file."""
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spread_path = data_dir / "spread.txt"
    _write_spread_file(spread_path, wavelengths=[1000.0, 1100.0], num_rows=3)

    positions, weights, wavelengths = load_spread_profile_file_spectroscopy("spread.txt", "NUV")

    assert np.allclose(positions, [0.0, 1.0, 2.0])
    assert weights.shape == (3, 2)
    assert np.allclose(wavelengths, [1000.0, 1100.0])


# Tests: load_spread_profile_file_spectroscopy
# Behavior: raises ValueError when spread file path does not exist
def test_load_spread_profile_file_spectroscopy_missing_file_raises(monkeypatch, tmp_path):
    """load_spread_profile_file_spectroscopy raises ValueError when file does not exist (non-empty filename)."""
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError):
        load_spread_profile_file_spectroscopy("nonexistent_spread.txt", "NUV")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: empty spread filename raises ValueError
def test_load_spread_profile_file_spectroscopy_empty_filename_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)

    with pytest.raises(ValueError):
        load_spread_profile_file_spectroscopy("", "NUV")

    with pytest.raises(ValueError):
        load_spread_profile_file_spectroscopy("   ", "NUV")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: raises ValueError when 'pixels' header line is absent
def test_load_spread_profile_file_spectroscopy_missing_pixels_header_raises(monkeypatch, tmp_path):
    """load_spread_profile_file_spectroscopy raises ValueError when 'pixels' header line is absent."""
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "spread.txt", "# comment\n0  0.1 0.2\n1  0.3 0.4\n")

    with pytest.raises(ValueError):
        load_spread_profile_file_spectroscopy("spread.txt", "NUV")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: raises ValueError when header wavelength count differs from weight columns
def test_load_spread_profile_file_spectroscopy_header_count_mismatch_raises(monkeypatch, tmp_path):
    """load_spread_profile_file_spectroscopy raises ValueError when header wavelength count != weight columns."""
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "spread.txt", "pixels 1000 1100 1200\n0  0.1 0.2\n1  0.3 0.4\n")

    with pytest.raises(ValueError):
        load_spread_profile_file_spectroscopy("spread.txt", "NUV")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: accepts whitespace/blank lines and preserves float32 output arrays
def test_load_spread_profile_file_spectroscopy_leading_trailing_whitespace_ok(monkeypatch, tmp_path):
    """Whitespace and blank lines do not break spread parsing; output dtypes are float."""
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(
        data_dir / "spread.txt",
        "   pixels   1000   1100   \n\n  0    0.10   0.20   \n  1    0.30   0.40   \n\n",
    )

    pos, w, wl = load_spread_profile_file_spectroscopy("spread.txt", "NUV")

    assert np.issubdtype(pos.dtype, np.floating)
    assert np.issubdtype(w.dtype, np.floating)
    assert np.issubdtype(wl.dtype, np.floating)
    assert pos.dtype == np.float32
    assert w.dtype == np.float32
    assert wl.dtype == np.float32
    assert wl[0] == pytest.approx(1000.0)
    assert wl[1] == pytest.approx(1100.0)
    assert w[1, 1] == pytest.approx(0.40)


# Tests: load_spread_profile_file_spectroscopy
# Behavior: falls back to UTF-16 when UTF-8 decode is not possible
def test_load_spread_profile_file_spectroscopy_encoding_fallback_utf16(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spread_path = data_dir / "spread.txt"
    text = "# comment\npixels  1000  1100\n0  0.5  0.5\n1  0.5  0.5\n2  0.5  0.5\n"
    spread_path.write_text(text, encoding="utf-16")

    positions, weights, wavelengths = load_spread_profile_file_spectroscopy("spread.txt", "NUV")

    assert np.allclose(positions, [0.0, 1.0, 2.0])
    assert weights.shape == (3, 2)
    assert np.allclose(wavelengths, [1000.0, 1100.0])


# Tests: load_spread_profile_file_spectroscopy
# Behavior: non-numeric data rows fail numeric table parsing
def test_load_spread_profile_file_spectroscopy_invalid_numeric_data_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "spread.txt"
    path.write_text(
        "pixels 1000 1100\n"
        "0 0.1 0.2\n"
        "bad row here\n"
        "1 0.3 0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Failed to parse numeric spread table"):
        load_spread_profile_file_spectroscopy("spread.txt", "VIS")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: numeric table with only one column (dy only) is rejected
def test_load_spread_profile_file_spectroscopy_dy_only_table_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "spread.txt"
    path.write_text(
        "pixels 1000\n"
        "0\n"
        "1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid spread table structure"):
        load_spread_profile_file_spectroscopy("spread.txt", "VIS")


# Tests: load_spread_profile_file_spectroscopy
# Behavior: single numeric row (1D loadtxt output) is rejected as invalid structure
def test_load_spread_profile_file_spectroscopy_single_numeric_row_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "spread.txt"
    path.write_text(
        "pixels 1000 1100\n"
        "0 0.1 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid spread table structure"):
        load_spread_profile_file_spectroscopy("spread.txt", "VIS")