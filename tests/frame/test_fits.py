import numpy as np
from astropy.io import fits
from types import SimpleNamespace

from frame.fits import write_fits_frames


def test_write_fits_frames(tmp_path):
    """write_fits_frames writes the correct number of FITS files with expected data and header entries."""
    ctx = SimpleNamespace(output_dir=tmp_path)

    # Prepare dummy data
    frames = [
        np.ones((5, 5)),
        np.full((5, 5), 2.0),
        np.full((5, 5), 3.0),
    ]

    headers = [
        fits.Header([("TESTKEY", "A", "Test header entry")]),
        fits.Header([("TESTKEY", "B", "Test header entry")]),
        fits.Header([("TESTKEY", "C", "Test header entry")]),
    ]

    frame_type = "DARK"
    channel_tag = "NUV"

    write_fits_frames(frames, headers, frame_type, channel_tag, ctx)

    # Check that 3 FITS files were written
    for k in range(3):
        filename = tmp_path / f"WALTzER_{channel_tag}_{frame_type}_{k:05d}.fits"
        assert filename.exists()

        # Open FITS file and verify contents
        with fits.open(filename) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header

            # Check data matches input
            assert np.allclose(data, frames[k])

            # Check header contains original key
            assert hdr["TESTKEY"] == headers[k]["TESTKEY"]

            # Check FILENAME was appended
            assert hdr["FILENAME"] == f"WALTzER_{channel_tag}_{frame_type}_{k:05d}.fits"


def test_write_fits_frames_empty_list_writes_nothing(tmp_path):
    """write_fits_frames with empty frames list returns early and writes no files."""
    ctx = SimpleNamespace(output_dir=tmp_path)

    write_fits_frames([], [], "DARK", "NUV", ctx)

    fits_files = list(tmp_path.glob("*.fits"))
    assert len(fits_files) == 0


def test_write_fits_frames_mismatched_lengths_uses_shorter(tmp_path):
    """When frames and headers have different lengths, zip truncates; only min(len) files are written."""
    ctx = SimpleNamespace(output_dir=tmp_path)

    frames = [np.ones((2, 2)), np.full((2, 2), 2.0)]
    headers = [fits.Header([("K", "1", "")])]  # only 1 header

    write_fits_frames(frames, headers, "BIAS", "VIS", ctx)

    # zip truncates to shorter length (1), so only 1 file written
    fits_files = list(tmp_path.glob("*.fits"))
    assert len(fits_files) == 1
    assert fits_files[0].name == "WALTzER_VIS_BIAS_00000.fits"
