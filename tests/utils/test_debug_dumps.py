# Covers FITS writing behavior for empty input and correct file creation for non empty frames and headers.
import numpy as np
from astropy.io import fits
from frame.fits import write_fits_frames


def test_write_fits_frames_no_frames_returns_without_creating_files(tmp_path):
    write_fits_frames([], [], "BIAS", "NUV", tmp_path)
    assert list(tmp_path.glob("*.fits")) == []


def test_write_fits_frames_writes_one_file_with_header(tmp_path):
    frame = np.zeros((2, 3), dtype=float)
    hdr = fits.Header()
    hdr["FILETYPE"] = "BIAS"

    write_fits_frames([frame], [hdr], "BIAS", "NUV", tmp_path)

    out = tmp_path / "WALTzER_NUV_BIAS_00000.fits"
    assert out.exists()

    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (2, 3)
        assert hdul[0].header["FILETYPE"] == "BIAS"


def test_write_fits_frames_zips_frames_and_headers_stops_at_shorter(tmp_path):
    frame0 = np.zeros((2, 2), dtype=float)
    frame1 = np.ones((2, 2), dtype=float)

    hdr0 = fits.Header()
    hdr0["IDX"] = 0

    write_fits_frames([frame0, frame1], [hdr0], "DARK", "VIS", tmp_path)

    assert (tmp_path / "WALTzER_VIS_DARK_00000.fits").exists()
    assert not (tmp_path / "WALTzER_VIS_DARK_00001.fits").exists()
