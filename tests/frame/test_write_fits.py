"""Tests for frame.write_fits.write_fits_frame (single-frame writer)."""

import numpy as np
from astropy.io import fits
from types import SimpleNamespace

from frame.write_fits import write_fits_frame


def _ctx(tmp_path, target_name="TestStar"):
    return SimpleNamespace(output_dir=tmp_path, target_name=target_name)


def test_write_fits_frame_writes_one_file(tmp_path):
    """Single frame and header produce one FITS file with correct data and header."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((2, 3), dtype=float)
    hdr = fits.Header()
    hdr["FILETYPE"] = "BIAS"

    write_fits_frame(frame, hdr, "BIAS", "NUV", ctx)

    out = tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (2, 3)
        assert np.allclose(hdul[0].data, frame)
        assert hdul[0].header["FILETYPE"] == "BIAS"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_NUV_BIAS_00000.fits"


def test_write_fits_frame_appends_filename_to_header(tmp_path):
    """write_fits_frame appends FILENAME card; does not mutate caller's header."""
    ctx = _ctx(tmp_path)
    frame = np.ones((1, 1))
    hdr = fits.Header([("ORIG", "x", "")])

    write_fits_frame(frame, hdr, "science", "VIS", ctx)

    assert hdr.get("FILENAME") is None  # original unchanged
    with fits.open(tmp_path / "WALTzER_TestStar_VIS_science_00000.fits") as hdul:
        assert hdul[0].header["ORIG"] == "x"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_VIS_science_00000.fits"


def test_write_fits_frame_index_in_filename(tmp_path):
    """index is used in output filename."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((1, 1))
    hdr = fits.Header()

    write_fits_frame(frame, hdr, "BIAS", "NUV", ctx, index=2)

    out = tmp_path / "WALTzER_TestStar_NUV_BIAS_00002.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_NUV_BIAS_00002.fits"


def test_write_fits_frame_overwrites_existing_file(tmp_path):
    """Second write with same filename overwrites (overwrite=True)."""
    ctx = _ctx(tmp_path)
    frame1 = np.ones((2, 2))
    frame2 = np.full((2, 2), 99.0)
    hdr = fits.Header()

    write_fits_frame(frame1, hdr, "BIAS", "NUV", ctx)
    write_fits_frame(frame2, fits.Header(), "BIAS", "NUV", ctx)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits") as hdul:
        assert np.allclose(hdul[0].data, frame2)
