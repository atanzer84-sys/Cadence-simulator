"""Thin integration tests for frame.write_fits.write_fits_frame."""

import numpy as np
from astropy.io import fits
from types import SimpleNamespace

from frame.write_fits import write_fits_frame


def _ctx(tmp_path, target_name="TestStar"):
    return SimpleNamespace(output_dir=tmp_path, target_name=target_name)


def test_single_frame_integration(tmp_path):
    """End-to-end: one array+header written, file exists with correct content."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((2, 3), dtype=float)
    hdr = fits.Header()
    hdr["FILETYPE"] = "BIAS"

    write_fits_frame(frame, hdr, "BIAS", "NUV", ctx)

    out = tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert np.allclose(hdul[0].data, frame)
        assert hdul[0].header["FILETYPE"] == "BIAS"


def test_multiple_frames_via_index_integration(tmp_path):
    """End-to-end: caller writes multiple files by calling write_fits_frame with index=0,1,2."""
    ctx = _ctx(tmp_path)
    for i in range(3):
        frame = np.full((1, 1), i, dtype=float)
        hdr = fits.Header([("IDX", i, "")])
        write_fits_frame(frame, hdr, "DARK", "VIS", ctx, index=i)

    for k in range(3):
        path = tmp_path / f"WALTzER_TestStar_VIS_DARK_{k:05d}.fits"
        assert path.exists()
        with fits.open(path) as hdul:
            assert hdul[0].header["IDX"] == k
