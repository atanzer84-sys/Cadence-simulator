import numpy as np
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from frame.frame import generate_Frames


class DummyCfg:
    """Minimal channel config; add attributes when frame.py or its callees start using them."""
    def __init__(self, name):
        self.channel_name = name
        self.x_pixels = 5
        self.y_pixels = 5
        self.ccd_gain = 2.0
        self.bias_offset = 10.0
        self.read_noise = 3.0
        self.dark_current_sigma = 0.5
        self.dark_noise = 1.0
        self.exposure_s = 10.0
        self.spread_half_height_pix = 1
        self.mode = 1
        self.spread_profile_file = "dummy.fits"


def _common_setup(tmp_path):
    """Shared setup for generate_Frames tests."""
    counts_nuv = np.array([1, 2, 3, 4, 5], dtype=float)
    counts_vis = np.array([2, 3, 4, 5, 6], dtype=float)
    nuv_cfg = DummyCfg("NUV")
    vis_cfg = DummyCfg("VIS")
    ctx = SimpleNamespace(output_dir=tmp_path, timestamp=datetime(2024, 1, 1, 12, 0, 0))
    star = MagicMock()
    return counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star


def test_generate_Frames_minimal(tmp_path):
    """generate_Frames orchestrates get_global_config, bias/dark/science generation, FITS writing; PNG skipped when disabled."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_dark_and_bias_png = False
    mock_global_cfg.write_science_frames_png = False

    with patch("frame.frame.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame.initialize_fits_header", return_value=[]) as mock_init_header, \
         patch("frame.frame.generate_bias_frames", return_value=([np.zeros((5, 5))], [[]])), \
         patch("frame.frame.generate_dark_frames", return_value=([np.zeros((5, 5))], [[]])), \
         patch("frame.frame.generate_science_frames", return_value=([np.ones((5, 5))], [[]])) as mock_science, \
         patch("frame.frame.write_fits_frames") as mock_write_fits, \
         patch("frame.frame.write_frames_png") as mock_write_png:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_init_header.assert_called_once_with(star, ctx.timestamp)
    assert mock_science.call_count == 2
    assert mock_write_fits.call_count == 6  # bias NUV/VIS, dark NUV/VIS, science NUV/VIS

    write_fits_calls = mock_write_fits.call_args_list
    assert write_fits_calls[0].args[2] == "bias" and write_fits_calls[0].args[3] == "NUV"
    assert write_fits_calls[1].args[2] == "bias" and write_fits_calls[1].args[3] == "VIS"
    assert write_fits_calls[2].args[2] == "dark" and write_fits_calls[2].args[3] == "NUV"
    assert write_fits_calls[3].args[2] == "dark" and write_fits_calls[3].args[3] == "VIS"
    assert write_fits_calls[4].args[2] == "science" and write_fits_calls[4].args[3] == "NUV"
    assert write_fits_calls[5].args[2] == "science" and write_fits_calls[5].args[3] == "VIS"
    for call in write_fits_calls:
        assert call.args[4] is ctx

    mock_write_png.assert_not_called()


def test_generate_Frames_skips_bias_dark_when_n_bias_and_darkframes_zero(tmp_path):
    """When n_bias_and_darkframes=0, bias/dark generation and FITS writing are skipped; science still runs."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 0
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_dark_and_bias_png = False
    mock_global_cfg.write_science_frames_png = False

    with patch("frame.frame.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame.initialize_fits_header", return_value=[]), \
         patch("frame.frame.generate_bias_frames") as mock_bias, \
         patch("frame.frame.generate_dark_frames") as mock_dark, \
         patch("frame.frame.generate_science_frames", return_value=([np.ones((5, 5))], [[]])) as mock_science, \
         patch("frame.frame.write_fits_frames") as mock_write_fits:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_bias.assert_not_called()
    mock_dark.assert_not_called()
    assert mock_science.call_count == 2
    assert mock_write_fits.call_count == 2  # science NUV, science VIS only


def test_generate_Frames_skips_science_when_n_science_frames_zero(tmp_path):
    """When n_science_frames_per_channel=0, science generation and FITS writing are skipped; bias/dark still run."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 0
    mock_global_cfg.write_dark_and_bias_png = False
    mock_global_cfg.write_science_frames_png = False

    with patch("frame.frame.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame.initialize_fits_header", return_value=[]), \
         patch("frame.frame.generate_bias_frames", return_value=([np.zeros((5, 5))], [[]])), \
         patch("frame.frame.generate_dark_frames", return_value=([np.zeros((5, 5))], [[]])), \
         patch("frame.frame.generate_science_frames") as mock_science, \
         patch("frame.frame.write_fits_frames") as mock_write_fits:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_science.assert_not_called()
    assert mock_write_fits.call_count == 4  # bias NUV/VIS, dark NUV/VIS only
