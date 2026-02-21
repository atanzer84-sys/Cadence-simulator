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

# Ensures zoom dumps only include wavelengths inside the defined debug windows.
def test_dump_1d_array_zoom_files_are_windowed(tmp_path):
    import numpy as np
    from utils import debug_dumps
    from utils.constants import DEBUG_WL_A_NUV

    wave = np.array([DEBUG_WL_A_NUV[0] - 1.0, DEBUG_WL_A_NUV[0], (DEBUG_WL_A_NUV[0] + DEBUG_WL_A_NUV[1]) / 2.0, DEBUG_WL_A_NUV[1], DEBUG_WL_A_NUV[1] + 1.0])
    values = np.arange(wave.size, dtype=float)

    debug_dumps.dump_1d_array(wave, values, tmp_path, "Star", "tag", full=False, zoom=True)

    data = np.loadtxt(tmp_path / "Star_tag_NUV.txt")
    assert data[:, 0].min() >= DEBUG_WL_A_NUV[0]
    assert data[:, 0].max() <= DEBUG_WL_A_NUV[1]
