"""
Direct tests for photometry low-level calibration file loaders.
"""

from pathlib import Path

import numpy as np
import pytest

from loaders.load_channel_files_photometry import load_psf_image_file

_REPO_ROOT_PHOT = "loaders.load_channel_files_photometry.get_repo_root"


def _write_psf_file_utf16(path: Path, grid: np.ndarray, center_x_1b: int | None = 1, center_y_1b: int | None = 1) -> None:
    """Write a minimal UTF-16 PSF file with optional center header and numeric grid."""
    lines: list[str] = ["Listing of Huygens PSF Data", ""]
    if center_x_1b is not None and center_y_1b is not None:
        lines.append(f"Center point is: {center_x_1b}, {center_y_1b}")
    lines.append("Values are relative intensity.")
    lines.append("")
    for row in grid:
        lines.append(" ".join(f"{float(v):.6e}" for v in row))
    path.write_text("\n".join(lines), encoding="utf-16")


# Tests: load_psf_image_file
# Behavior: reads PSF, normalizes, and parses header center
def test_load_psf_image_file_success_with_center_header(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    grid[1, 2] = 4.0
    grid[1, 3] = 1.0
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=4, center_y_1b=2)

    psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR")

    assert psf.shape == (4, 20)
    assert float(np.sum(psf)) == pytest.approx(1.0)
    assert (cy, cx) == (1, 3)


# Tests: load_psf_image_file
# Behavior: missing file raises ValueError
def test_load_psf_image_file_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError):
        load_psf_image_file("missing_psf.txt", "NIR")


# Tests: load_psf_image_file
# Behavior: zero-sum PSF raises ValueError
def test_load_psf_image_file_invalid_sum_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=1, center_y_1b=1)

    with pytest.raises(ValueError):
        load_psf_image_file("nir_psf.txt", "NIR")


# Tests: load_psf_image_file
# Behavior: empty filename raises ValueError
def test_load_psf_image_file_empty_filename_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)

    with pytest.raises(ValueError):
        load_psf_image_file("", "NIR")

    with pytest.raises(ValueError):
        load_psf_image_file("   ", "NIR")


# Tests: load_psf_image_file
# Behavior: falls back to peak when header missing
def test_load_psf_image_file_no_center_header_falls_back_to_peak(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    grid[2, 5] = 7.0
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=None, center_y_1b=None)

    _psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR")

    assert (cy, cx) == (2, 5)


# Tests: load_psf_image_file
# Behavior: missing numeric grid raises ValueError
def test_load_psf_image_file_no_numeric_grid_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "nir_psf.txt").write_text(
        "Listing of Huygens PSF Data\nCenter point is: 1, 1\nValues are relative intensity.\nno numeric grid here\n",
        encoding="utf-16",
    )

    with pytest.raises(ValueError):
        load_psf_image_file("nir_psf.txt", "NIR")


# Tests: load_psf_image_file
# Behavior: normalizes uniform grid correctly
def test_load_psf_image_file_normalization(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.ones((4, 20), dtype=float)
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid)

    psf, _, _ = load_psf_image_file("nir_psf.txt", "NIR")

    assert np.allclose(psf, np.full((4, 20), 1.0 / 80.0))


# Tests: load_psf_image_file
# Behavior: stops grid parsing on irregular row length
def test_load_psf_image_file_stops_on_irregular_row(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "nir_psf.txt"
    lines = [
        "Listing of Huygens PSF Data",
        "Values are relative intensity.",
        "",
        " ".join(["1.0"] * 20),
        " ".join(["2.0"] * 20),
        " ".join(["3.0"] * 20),
        " ".join(["4.0"] * 20),
        " ".join(["5.0"] * 10),
    ]
    path.write_text("\n".join(lines), encoding="utf-16")

    psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR")

    assert psf.shape == (4, 20)
    assert (cy, cx) == (3, 0)
    assert float(np.sum(psf)) == pytest.approx(1.0)


# Tests: load_psf_image_file
# Behavior: default min_cols=20 rejects 19-column grid, override min_cols=19 accepts it
def test_load_psf_image_file_min_cols_default_and_override(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "nir_psf.txt"
    lines = [
        "Listing of Huygens PSF Data",
        "Center point is: 1, 1",
        "Values are relative intensity.",
        "",
    ]
    lines.extend([" ".join(["1.0"] * 19) for _ in range(4)])
    path.write_text("\n".join(lines), encoding="utf-16")

    with pytest.raises(ValueError):
        load_psf_image_file("nir_psf.txt", "NIR")

    psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR", min_cols=19)
    assert psf.shape == (4, 19)
    assert (cy, cx) == (0, 0)
    assert float(np.sum(psf)) == pytest.approx(1.0)


# Tests: load_psf_image_file
# Behavior: default stability_rows=3 requires four stable rows; override can relax requirement
def test_load_psf_image_file_stability_rows_default_and_override(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "nir_psf.txt"
    lines = [
        "Listing of Huygens PSF Data",
        "Center point is: 1, 1",
        "Values are relative intensity.",
        "",
        " ".join(["1.0"] * 20),
        " ".join(["2.0"] * 20),
        " ".join(["3.0"] * 20),
        " ".join(["4.0"] * 10),  # breaks the 4th stable row requirement for default stability_rows=3
    ]
    path.write_text("\n".join(lines), encoding="utf-16")

    with pytest.raises(ValueError):
        load_psf_image_file("nir_psf.txt", "NIR")

    psf, _, _ = load_psf_image_file("nir_psf.txt", "NIR", stability_rows=2)
    assert psf.shape == (3, 20)
    assert float(np.sum(psf)) == pytest.approx(1.0)


# Tests: load_psf_image_file
# Behavior: falls back to UTF-8 when UTF-16 decode is not possible
def test_load_psf_image_file_encoding_fallback_utf8(monkeypatch, tmp_path):
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = data_dir / "nir_psf.txt"
    lines = [
        "Listing of Huygens PSF Data",
        "Center point is: 1, 1",
        "Values are relative intensity.",
        "",
    ]
    lines.extend([" ".join(["1.0"] * 20) for _ in range(4)])
    path.write_text("\n".join(lines), encoding="utf-8")

    psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR")
    assert psf.shape == (4, 20)
    assert (cy, cx) == (0, 0)