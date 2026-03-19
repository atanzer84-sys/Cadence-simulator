"""
Direct tests for photometry low-level calibration file loaders.
"""

from pathlib import Path

import numpy as np
import pytest

from loaders.load_channel_files_photometry import load_psf_image_file

_REPO_ROOT_PHOT = "loaders.load_channel_files_photometry.get_repo_root"


def _assert_error_contains(exc: BaseException, *needles: str) -> None:
    """Assert a raised error message contains stable keyword fragments."""
    msg = str(exc).lower()
    for needle in needles:
        assert needle.lower() in msg


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


def test_load_psf_image_file_success_with_center_header(monkeypatch, tmp_path):
    """load_psf_image_file reads UTF-16 grid, normalizes to sum=1, and parses header center."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    grid[1, 2] = 4.0
    grid[1, 3] = 1.0
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=4, center_y_1b=2)

    psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR", None)

    assert psf.shape == (4, 20)
    assert float(np.sum(psf)) == pytest.approx(1.0)
    assert (cy, cx) == (1, 3)


def test_load_psf_image_file_missing_file_raises(monkeypatch, tmp_path):
    """Missing PSF image file raises ValueError."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    with pytest.raises(ValueError) as exc:
        load_psf_image_file("missing_psf.txt", "NIR", None)
    _assert_error_contains(exc.value, "psf image file", "not found")


def test_load_psf_image_file_invalid_sum_raises(monkeypatch, tmp_path):
    """All-zero PSF grid raises ValueError due to invalid normalization sum."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=1, center_y_1b=1)

    with pytest.raises(ValueError) as exc:
        load_psf_image_file("nir_psf.txt", "NIR", None)
    _assert_error_contains(exc.value, "psf", "sum", "invalid")


def test_load_psf_image_file_empty_filename_raises_not_configured(monkeypatch, tmp_path):
    """Empty/whitespace filename raises not-configured ValueError."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)

    with pytest.raises(ValueError) as exc_empty:
        load_psf_image_file("", "NIR", None)
    _assert_error_contains(exc_empty.value, "psf image file", "not configured")

    with pytest.raises(ValueError) as exc_ws:
        load_psf_image_file("   ", "NIR", None)
    _assert_error_contains(exc_ws.value, "psf image file", "not configured")


def test_load_psf_image_file_no_center_header_falls_back_to_peak(monkeypatch, tmp_path):
    """When center header is missing, center falls back to numeric peak location."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    grid = np.zeros((4, 20), dtype=float)
    grid[2, 5] = 7.0
    _write_psf_file_utf16(data_dir / "nir_psf.txt", grid, center_x_1b=None, center_y_1b=None)

    _psf, cy, cx = load_psf_image_file("nir_psf.txt", "NIR", None)

    assert (cy, cx) == (2, 5)


def test_load_psf_image_file_no_numeric_grid_raises(monkeypatch, tmp_path):
    """File with no stable numeric grid raises a clear ValueError."""
    monkeypatch.setattr(_REPO_ROOT_PHOT, lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nir_psf.txt").write_text(
        "Listing of Huygens PSF Data\nCenter point is: 1, 1\nValues are relative intensity.\nno numeric grid here\n",
        encoding="utf-16",
    )

    with pytest.raises(ValueError) as exc:
        load_psf_image_file("nir_psf.txt", "NIR", None)
    _assert_error_contains(exc.value, "could not locate", "numeric grid")
