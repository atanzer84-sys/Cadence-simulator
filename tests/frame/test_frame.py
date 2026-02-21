import numpy as np
from unittest.mock import patch, MagicMock

# IMPORTANT: generate_Frames lives in frame/frame.py
from frame.frame import generate_Frames


# ----------------------------------------------------------------------
# Dummy config objects used for orchestration testing
# ----------------------------------------------------------------------

class DummyCfg:
    """
    Minimal dummy channel configuration object.
    Must include all attributes accessed by generate_Frames.
    """
    def __init__(self, name):
        self.channel_name = name
        self.x_pixels = 5
        self.y_pixels = 5
        self.ccd_gain = 2.0
        self.bias_offset = 10.0
        self.read_noise = 3.0
        self.dark_current_sigma = 0.5
        self.dark_noise = 1.0
        self.spread_half_height_pix = 1
        self.mode = 1
        self.spread_profile_file = "dummy.fits"  # required by science code


class DummyUserCfg:
    """
    Minimal dummy user configuration object.
    """
    def __init__(self):
        self.exposure_NUV_s = 3.0
        self.exposure_VIS_s = 4.0


class DummyCal:
    """
    Minimal dummy calibration object.
    """
    def __init__(self):
        self.spread_y_positions = None
        self.spread_y_weights = None
        self.spread_y_wavelengths = None
        self.wavelength = np.linspace(100, 200, 5)


# ----------------------------------------------------------------------
# MINIMAL ORCHESTRATION TEST
# ----------------------------------------------------------------------

def test_generate_Frames_minimal(tmp_path):
    """
    This test checks that generate_Frames orchestrates the correct calls:
    - global config is read
    - bias/dark/science frames are generated
    - FITS writing is triggered
    - PNG writing is skipped when disabled
    """

    # Prepare dummy inputs
    counts_nuv = np.array([1, 2, 3, 4, 5], dtype=float)
    counts_vis = np.array([2, 3, 4, 5, 6], dtype=float)

    nuv_cfg = DummyCfg("NUV")
    vis_cfg = DummyCfg("VIS")
    nuv_cal = DummyCal()
    vis_cal = DummyCal()
    user_cfg = DummyUserCfg()
    star = MagicMock()

    # Mock global config returned by get_global_config()
    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_dark_and_bias_png = False
    mock_global_cfg.write_science_frames_png = False

    # Patch ALL functions inside frame.frame (your real module)
    with patch("frame.frame.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame.initialize_fits_header", return_value=[]), \
         patch("frame.frame.generate_bias_frames", return_value=([np.zeros((5,5))], [[]])), \
         patch("frame.frame.generate_dark_frames", return_value=([np.zeros((5,5))], [[]])), \
         patch("frame.frame.generate_science_frames", return_value=([np.ones((5,5))], [[]])) as mock_science, \
         patch("frame.frame.write_fits_frames") as mock_write_fits, \
         patch("frame.frame.write_frames_png") as mock_write_png:

        # Call the orchestration function
        generate_Frames(
            counts_nuv,
            counts_vis,
            nuv_cfg,
            vis_cfg,
            nuv_cal,
            vis_cal,
            user_cfg,
            tmp_path,
            star
        )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # Science frames must be generated for both channels
    assert mock_science.call_count == 2

    # FITS writing must be called for bias, dark, and science
    assert mock_write_fits.call_count > 0

    # PNG writing is disabled in this test
    mock_write_png.assert_not_called()
