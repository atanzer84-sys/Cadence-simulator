"""Tests for frame.fits.write_fits_frames."""

import numpy as np
from astropy.io import fits
from types import SimpleNamespace

from frame.fits import write_fits_frames


def _ctx(tmp_path, target_name="TestStar"):
    return SimpleNamespace(output_dir=tmp_path, target_name=target_name)


# --- Empty / early return ---


def test_write_fits_empty_frames_returns_without_creating_files(tmp_path):
    """Empty frames list returns early; no FITS files created."""
    write_fits_frames([], [], "BIAS", "NUV", _ctx(tmp_path))
    assert list(tmp_path.glob("*.fits")) == []


def test_write_fits_empty_headers_with_frames_truncates_to_zero(tmp_path):
    """zip(frames, []) yields no pairs; no files written."""
    frame = np.zeros((2, 2))
    write_fits_frames([frame], [], "DARK", "VIS", _ctx(tmp_path))
    assert list(tmp_path.glob("*.fits")) == []


# --- Single frame ---


def test_write_fits_single_frame_writes_one_file(tmp_path):
    """Single frame and header produce one FITS file with correct data and header."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((2, 3), dtype=float)
    hdr = fits.Header()
    hdr["FILETYPE"] = "BIAS"

    write_fits_frames([frame], [hdr], "BIAS", "NUV", ctx)

    out = tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (2, 3)
        assert np.allclose(hdul[0].data, frame)
        assert hdul[0].header["FILETYPE"] == "BIAS"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_NUV_BIAS_00000.fits"


def test_write_fits_appends_filename_to_header(tmp_path):
    """write_fits_frames appends FILENAME card to each header."""
    ctx = _ctx(tmp_path)
    frame = np.ones((1, 1))
    hdr = fits.Header([("ORIG", "x", "")])

    write_fits_frames([frame], [hdr], "science", "VIS", ctx)

    with fits.open(tmp_path / "WALTzER_TestStar_VIS_science_00000.fits") as hdul:
        assert hdul[0].header["ORIG"] == "x"
        assert hdul[0].header["FILENAME"] == "WALTzER_TestStar_VIS_science_00000.fits"


# --- Multiple frames ---


def test_write_fits_multiple_frames_writes_all(tmp_path):
    """Multiple frames produce one FITS file each with correct index in filename."""
    ctx = _ctx(tmp_path)
    frames = [
        np.full((2, 2), 1.0),
        np.full((2, 2), 2.0),
        np.full((2, 2), 3.0),
    ]
    headers = [fits.Header([("IDX", i, "")]) for i in range(3)]

    write_fits_frames(frames, headers, "DARK", "NUV", ctx)

    for k in range(3):
        path = tmp_path / f"WALTzER_TestStar_NUV_DARK_{k:05d}.fits"
        assert path.exists()
        with fits.open(path) as hdul:
            assert np.allclose(hdul[0].data, frames[k])
            assert hdul[0].header["IDX"] == k


# --- Zip truncation (mismatched lengths) ---


def test_write_fits_more_frames_than_headers_truncates(tmp_path):
    """When frames > headers, zip truncates; only min(len) files written."""
    ctx = _ctx(tmp_path)
    frames = [np.ones((2, 2)), np.full((2, 2), 2.0)]
    headers = [fits.Header([("K", "1", "")])]

    write_fits_frames(frames, headers, "BIAS", "VIS", ctx)

    assert (tmp_path / "WALTzER_TestStar_VIS_BIAS_00000.fits").exists()
    assert not (tmp_path / "WALTzER_TestStar_VIS_BIAS_00001.fits").exists()


def test_write_fits_more_headers_than_frames_truncates(tmp_path):
    """When headers > frames, zip truncates; only min(len) files written."""
    ctx = _ctx(tmp_path)
    frames = [np.ones((2, 2))]
    headers = [
        fits.Header([("A", "1", "")]),
        fits.Header([("B", "2", "")]),
    ]

    write_fits_frames(frames, headers, "DARK", "IR", ctx)

    assert (tmp_path / "WALTzER_TestStar_IR_DARK_00000.fits").exists()
    assert not (tmp_path / "WALTzER_TestStar_IR_DARK_00001.fits").exists()


# --- Overwrite ---


def test_write_fits_overwrites_existing_file(tmp_path):
    """Second write with same filename overwrites (overwrite=True)."""
    ctx = _ctx(tmp_path)
    frame1 = np.ones((2, 2))
    frame2 = np.full((2, 2), 99.0)
    hdr = fits.Header()

    write_fits_frames([frame1], [hdr], "BIAS", "NUV", ctx)
    write_fits_frames([frame2], [hdr.copy()], "BIAS", "NUV", ctx)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits") as hdul:
        assert np.allclose(hdul[0].data, frame2)


# --- Channel and frame type variants ---


def test_write_fits_channel_tags_nuv_vis_ir(tmp_path):
    """Filename includes channel_tag (NUV, VIS, IR)."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((1, 1))
    hdr = fits.Header()

    for tag in ("NUV", "VIS", "IR"):
        write_fits_frames([frame], [hdr.copy()], "bias", tag, ctx)

    assert (tmp_path / "WALTzER_TestStar_NUV_bias_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_VIS_bias_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_IR_bias_00000.fits").exists()


def test_write_fits_frame_types_bias_dark_science(tmp_path):
    """Filename includes frame_type (bias, dark, science)."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((1, 1))
    hdr = fits.Header()

    for ftype in ("bias", "dark", "science"):
        write_fits_frames([frame], [hdr.copy()], ftype, "NUV", ctx)

    assert (tmp_path / "WALTzER_TestStar_NUV_bias_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_NUV_dark_00000.fits").exists()
    assert (tmp_path / "WALTzER_TestStar_NUV_science_00000.fits").exists()


# --- Data preservation ---


def test_write_fits_preserves_float_dtype(tmp_path):
    """Float array data is preserved in FITS."""
    ctx = _ctx(tmp_path)
    frame = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=float)
    hdr = fits.Header()

    write_fits_frames([frame], [hdr], "BIAS", "NUV", ctx)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_BIAS_00000.fits") as hdul:
        assert np.allclose(hdul[0].data, frame)


def test_write_fits_preserves_header_with_comment(tmp_path):
    """Header cards with comment are preserved."""
    ctx = _ctx(tmp_path)
    frame = np.zeros((1, 1))
    hdr = fits.Header([("EXPTIME", 10.0, "Exposure time [s]")])

    write_fits_frames([frame], [hdr], "DARK", "NUV", ctx)

    with fits.open(tmp_path / "WALTzER_TestStar_NUV_DARK_00000.fits") as hdul:
        assert hdul[0].header["EXPTIME"] == 10.0
