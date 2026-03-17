"""Tests for frame.write_fits.write_fits_frame."""

import numpy as np
from astropy.io import fits
from types import SimpleNamespace

from frame.frame_class import Frame
from frame.write_fits import write_fits_frame


def _ctx(tmp_path, target_name="TestStar"):
    return SimpleNamespace(output_dir=tmp_path, target_name=target_name)


def _frame(data, header, frame_type, channel_tag):
    return Frame(data=data, header=header, frame_type=frame_type, channel_tag=channel_tag)


# --- Empty / early return ---


def test_write_fits_empty_frames_returns_without_creating_files(tmp_path):
    """No frames written when none are passed (no call to write_fits_frame)."""
    # API is single-frame; "empty" means we never call write_fits_frame
    assert list(tmp_path.glob("*.fits")) == []


def test_write_fits_empty_headers_with_frames_truncates_to_zero(tmp_path):
    """With single-frame API, no call means no files written."""
    _ctx(tmp_path)
    # No write_fits_frame call → no files
    assert list(tmp_path.glob("*.fits")) == []


# --- Single frame ---


def test_write_fits_single_frame_writes_one_file(tmp_path):
    """Single frame and header produce one FITS file with correct data and header."""
    ctx = _ctx(tmp_path)
    data = np.zeros((2, 3), dtype=float)
    hdr = fits.Header()
    hdr["FILETYPE"] = "BIAS"
    f = _frame(data, hdr, "BIAS", "NUV")
    exposure = 10.0
    write_fits_frame(f, ctx, 0, exposure)

    out = tmp_path / "WALTzER_TestStar_NUV_BIAS_10s_00000.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (2, 3)
        assert np.allclose(hdul[0].data, data)
        assert hdul[0].header["FILETYPE"] == "BIAS"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_NUV_BIAS_10s_00000.fits"


def test_write_fits_appends_filename_to_header(tmp_path):
    """write_fits_frame appends FILENAME card to header."""
    ctx = _ctx(tmp_path)
    data = np.ones((1, 1))
    hdr = fits.Header([("ORIG", "x", "")])
    f = _frame(data, hdr, "science", "VIS")
    exposure = 20.0
    write_fits_frame(f, ctx, 0, exposure)

    with fits.open(tmp_path / "WALTzER_TestStar_VIS_science_20s_00000.fits") as hdul:
        assert hdul[0].header["ORIG"] == "x"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_VIS_science_20s_00000.fits"


# --- Multiple frames ---


def test_write_fits_multiple_frames_writes_all(tmp_path):
    """Multiple frames produce one FITS file each with correct index in filename."""
    ctx = _ctx(tmp_path)
    data_list = [
        np.full((2, 2), 1.0),
        np.full((2, 2), 2.0),
        np.full((2, 2), 3.0),
    ]
    headers = [fits.Header([("IDX", i, "")]) for i in range(3)]
    exposure = 30.0

    for k in range(3):
        f = _frame(data_list[k], headers[k], "DARK", "NUV")
        write_fits_frame(f, ctx, k, exposure)

    for k in range(3):
        path = tmp_path / f"WALTzER_TestStar_NUV_DARK_30s_{k:05d}.fits"
        assert path.exists()
        with fits.open(path) as hdul:
            assert np.allclose(hdul[0].data, data_list[k])
            assert hdul[0].header["IDX"] == k


# --- Zip truncation (mismatched lengths) ---


def test_write_fits_more_frames_than_headers_truncates(tmp_path):
    """When writing only one frame (one header), only one file is created."""
    ctx = _ctx(tmp_path)
    f = _frame(np.ones((2, 2)), fits.Header([("K", "1", "")]), "BIAS", "VIS")
    exposure = 40.0
    write_fits_frame(f, ctx, 0, exposure)

    assert (tmp_path / "WALTzER_TestStar_VIS_BIAS_40s_00000.fits").exists()
    assert not (tmp_path / "WALTzER_TestStar_VIS_BIAS_40s_00001.fits").exists()


def test_write_fits_more_headers_than_frames_truncates(tmp_path):
    """When writing only one frame, only one file is created."""
    ctx = _ctx(tmp_path)
    f = _frame(np.ones((2, 2)), fits.Header([("A", "1", "")]), "DARK", "NIR")
    exposure = 50.0
    write_fits_frame(f, ctx, 0, exposure)

    assert (tmp_path / "WALTzER_TestStar_NIR_DARK_50s_00000.fits").exists()
    assert not (tmp_path / "WALTzER_TestStar_NIR_DARK_50s_00001.fits").exists()


# --- Overwrite ---


def test_write_fits_overwrites_existing_file(tmp_path):
    """Second write with same filename overwrites (overwrite=True)."""
    ctx = _ctx(tmp_path)
    frame1 = np.ones((2, 2))
    frame2 = np.full((2, 2), 99.0)
    hdr = fits.Header()
    exposure = 60.0

    write_fits_frame(_frame(frame1, hdr, "BIAS", "NUV"), ctx, 0, exposure)
    write_fits_frame(_frame(frame2, hdr.copy(), "BIAS", "NUV"), ctx, 0, exposure)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_BIAS_60s_00000.fits") as hdul:
        assert np.allclose(hdul[0].data, frame2)


# --- Channel and frame type variants ---


def test_write_fits_channel_tags_nuv_vis_ir(tmp_path):
    """Filename includes channel_tag (NUV, VIS, IR)."""
    ctx = _ctx(tmp_path)
    data = np.zeros((1, 1))
    hdr = fits.Header()
    exposure = 70.0

    for tag in ("NUV", "VIS", "NIR"):
        f = _frame(data, hdr.copy(), "bias", tag)
        write_fits_frame(f, ctx, 0, exposure)

    assert (tmp_path / "WALTzER_TestStar_NUV_bias_70s_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_VIS_bias_70s_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_NIR_bias_70s_00000.fits").exists()


def test_write_fits_frame_types_bias_dark_science(tmp_path):
    """Filename includes frame_type (bias, dark, science)."""
    ctx = _ctx(tmp_path)
    data = np.zeros((1, 1))
    hdr = fits.Header()
    exposure = 80.0

    for ftype in ("bias", "dark", "science"):
        f = _frame(data, hdr.copy(), ftype, "NUV")
        write_fits_frame(f, ctx, 0, exposure)

    assert (tmp_path / "WALTzER_TestStar_NUV_bias_80s_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_NUV_dark_80s_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_NUV_science_80s_00000.fits").exists()


# --- Data preservation ---


def test_write_fits_preserves_float_dtype(tmp_path):
    """Float array data is preserved in FITS."""
    ctx = _ctx(tmp_path)
    data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=float)
    hdr = fits.Header()
    exposure = 90.0
    write_fits_frame(_frame(data, hdr, "BIAS", "NUV"), ctx, 0, exposure)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_BIAS_90s_00000.fits") as hdul:
        assert np.allclose(hdul[0].data, data)


def test_write_fits_preserves_header_with_comment(tmp_path):
    """Header cards with comment are preserved."""
    ctx = _ctx(tmp_path)
    data = np.zeros((1, 1))
    hdr = fits.Header([("EXPTIME", 10.0, "Exposure time [s]")])
    exposure = 10.0
    write_fits_frame(_frame(data, hdr, "DARK", "NUV"), ctx, 0, exposure)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_DARK_10s_00000.fits") as hdul:
        assert hdul[0].header["EXPTIME"] == 10.0
