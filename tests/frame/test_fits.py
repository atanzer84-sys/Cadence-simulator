import numpy as np
from astropy.io import fits
from frame.fits import write_fits_frames


def test_write_fits_frames(tmp_path):
    # This test checks that write_fits_frames writes the correct number of FITS files
    # and that each file contains the expected data and header entries.

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

    # Call the function
    write_fits_frames(frames, headers, frame_type, channel_tag, tmp_path)

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
