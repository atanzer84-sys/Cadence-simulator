from pathlib import Path
import numpy as np
import pytest

from loaders.load_channel_files_spectroscopy import (
    load_polarization_file,
    load_spread_profile_file_spectroscopy,
    validate_polarization_config,
)

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
# Behavior: empty spread filename is a valid Gaussian-fallback case and returns all None
def test_load_spread_profile_file_spectroscopy_empty_filename_returns_none_triplet(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)

    assert load_spread_profile_file_spectroscopy("", "NUV") == (None, None, None)
    assert load_spread_profile_file_spectroscopy("   ", "NUV") == (None, None, None)


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


# Tests: load_polarization_file
# Behavior: empty or blank filename returns none pair
def test_load_polarization_file_empty_filename_returns_none_none(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)

    assert load_polarization_file("", "VIS") == (None, None)
    assert load_polarization_file("   ", "NUV") == (None, None)


# Tests: load_polarization_file
# Behavior: loads wavelength and delta columns as float32
def test_load_polarization_file_success(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "pol.txt", "1000  0.0\n1100  0.5\n1200  1.0\n")

    wl, delta = load_polarization_file("pol.txt", "VIS")

    assert wl.dtype == np.float32
    assert delta.dtype == np.float32
    assert np.allclose(wl, [1000.0, 1100.0, 1200.0])
    assert np.allclose(delta, [0.0, 0.5, 1.0])


# Tests: load_polarization_file
# Behavior: missing file raises ValueError
def test_load_polarization_file_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="Polarization file not found"):
        load_polarization_file("missing_pol.txt", "VIS")


# Tests: load_polarization_file
# Behavior: fewer than two columns raises ValueError
def test_load_polarization_file_one_column_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "pol.txt", "1000\n1100\n")

    with pytest.raises(ValueError, match="Invalid polarization file format"):
        load_polarization_file("pol.txt", "NUV")


# Tests: load_polarization_file
# Behavior: non-numeric content raises ValueError
def test_load_polarization_file_malformed_numeric_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "pol.txt", "1000  0.1\n1100  BAD\n")

    with pytest.raises(ValueError, match="Failed to parse polarization file"):
        load_polarization_file("pol.txt", "VIS")


# Tests: load_polarization_file
# Behavior: delta below zero raises ValueError
def test_load_polarization_file_delta_negative_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "pol.txt", "1000  -0.01\n1100  0.5\n")

    with pytest.raises(ValueError, match="polarization delta must be in range"):
        load_polarization_file("pol.txt", "VIS")


# Tests: load_polarization_file
# Behavior: delta above one raises ValueError
def test_load_polarization_file_delta_above_one_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_SPEC, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write(data_dir / "pol.txt", "1000  1.01\n1100  0.5\n")

    with pytest.raises(ValueError, match="polarization delta must be in range"):
        load_polarization_file("pol.txt", "VIS")


# Tests: validate_polarization_config
# Behavior: spectroscopy mode accepts missing polarization data
def test_validate_polarization_config_spectroscopy_allows_missing_polarization():
    validate_polarization_config("VIS", "spectroscopy", None, None, 0, 32)


# Tests: validate_polarization_config
# Behavior: spectropolarimetry requires polarization arrays
def test_validate_polarization_config_spectropolarimetry_requires_polarization():
    with pytest.raises(ValueError, match="spectropolarimetry requires polarization_file"):
        validate_polarization_config("VIS", "spectropolarimetry", None, None, 2, 32)


# Tests: validate_polarization_config
# Behavior: spectropolarimetry requires beam_separation_pix >= 1
def test_validate_polarization_config_spectropolarimetry_beam_separation_zero_raises():
    wl = np.array([1000.0], dtype=np.float32)
    d = np.array([0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="beam_separation_pix must be >= 1"):
        validate_polarization_config("NUV", "spectropolarimetry", wl, d, 0, 32)


# Tests: validate_polarization_config
# Behavior: spectropolarimetry rejects beam_separation_pix >= y_pixels
def test_validate_polarization_config_spectropolarimetry_beam_too_large_raises():
    wl = np.array([1000.0], dtype=np.float32)
    d = np.array([0.5], dtype=np.float32)
    with pytest.raises(ValueError, match=r"beam_separation_pix=6 too large for detector height=6"):
        validate_polarization_config("VIS", "spectropolarimetry", wl, d, 6, 6)


# Tests: validate_polarization_config
# Behavior: spectropolarimetry with valid inputs does not raise
def test_validate_polarization_config_spectropolarimetry_valid_ok():
    wl = np.array([1000.0, 1100.0], dtype=np.float32)
    d = np.array([0.2, 0.8], dtype=np.float32)
    validate_polarization_config("VIS", "spectropolarimetry", wl, d, 2, 64)


# Tests: validate_polarization_config
# Behavior: unknown observation_mode raises ValueError
def test_validate_polarization_config_invalid_mode_raises():
    with pytest.raises(ValueError, match=r"invalid observation_mode=imaging"):
        validate_polarization_config("VIS", "imaging", None, None, 0, 32)