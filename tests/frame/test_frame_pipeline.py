import numpy as np
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from frame.frame_pipeline import generate_frames


class DummyCfg:
    """Minimal channel config for frame_pipeline tests."""

    def __init__(self, name: str):
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
    counts_nuv = np.array([1, 2, 3, 4, 5], dtype=float)
    counts_vis = np.array([2, 3, 4, 5, 6], dtype=float)
    nuv_cfg = DummyCfg("NUV")
    vis_cfg = DummyCfg("VIS")
    ctx = SimpleNamespace(output_dir=tmp_path, timestamp=datetime(2024, 1, 1, 12, 0, 0))
    star = MagicMock()
    return counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star


def test_generate_frames_minimal_uses_fits_and_respects_png_flags(tmp_path):
    """generate_frames orchestrates bias/dark/science and delegates to write helpers; PNG flags respected."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_non_science_frames = 2
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_non_science_frames_png = True
    mock_global_cfg.write_science_frames_png = False

    with patch("frame.frame_pipeline.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame_pipeline.initialize_fits_header", return_value=[]) as mock_init_header, \
         patch("frame.frame_pipeline.generate_bias_frames", side_effect=[["bias_nuv"], ["bias_vis"]]) as mock_bias, \
         patch("frame.frame_pipeline.generate_dark_frames", side_effect=[["dark_nuv"], ["dark_vis"]]) as mock_dark, \
         patch("frame.frame_pipeline.generate_science_frames", side_effect=[["sci_nuv"], ["sci_vis"]]) as mock_science, \
         patch("frame.frame_pipeline._write_fits_for_all") as mock_write_fits_all, \
         patch("frame.frame_pipeline._write_png_for_all") as mock_write_png_all:

        generate_frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_init_header.assert_called_once_with(star, ctx.timestamp)
    assert mock_bias.call_count == 2
    assert mock_dark.call_count == 2
    assert mock_science.call_count == 2

    # One call for bias/dark, one for science
    assert mock_write_fits_all.call_count == 2

    # PNG only for bias/dark in this configuration
    mock_write_png_all.assert_called_once()


def test_generate_frames_skips_bias_dark_when_n_non_science_frames_zero(tmp_path):
    """When n_non_science_frames=0, bias/dark generation and FITS/PNG for them are skipped; science still runs."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_non_science_frames = 0
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_non_science_frames_png = True
    mock_global_cfg.write_science_frames_png = False

    with patch("frame.frame_pipeline.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame_pipeline.initialize_fits_header", return_value=[]), \
         patch("frame.frame_pipeline.generate_bias_frames") as mock_bias, \
         patch("frame.frame_pipeline.generate_dark_frames") as mock_dark, \
         patch("frame.frame_pipeline.generate_science_frames", side_effect=[["sci_nuv"], ["sci_vis"]]) as mock_science, \
         patch("frame.frame_pipeline._write_fits_for_all") as mock_write_fits_all, \
         patch("frame.frame_pipeline._write_png_for_all") as mock_write_png_all:

        generate_frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_bias.assert_not_called()
    mock_dark.assert_not_called()
    assert mock_science.call_count == 2

    # Only science FITS are written
    mock_write_fits_all.assert_called_once()

    # No PNG at all in this configuration
    mock_write_png_all.assert_not_called()


def test_generate_frames_skips_science_when_n_science_frames_zero(tmp_path):
    """When n_science_frames_per_channel=0, science generation and FITS/PNG for science are skipped; bias/dark still run."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_non_science_frames = 2
    mock_global_cfg.n_science_frames_per_channel = 0
    mock_global_cfg.write_non_science_frames_png = False
    mock_global_cfg.write_science_frames_png = True

    with patch("frame.frame_pipeline.get_global_config", return_value=mock_global_cfg), \
         patch("frame.frame_pipeline.initialize_fits_header", return_value=[]), \
         patch("frame.frame_pipeline.generate_bias_frames", side_effect=[["bias_nuv"], ["bias_vis"]]) as mock_bias, \
         patch("frame.frame_pipeline.generate_dark_frames", side_effect=[["dark_nuv"], ["dark_vis"]]) as mock_dark, \
         patch("frame.frame_pipeline.generate_science_frames") as mock_science, \
         patch("frame.frame_pipeline._write_fits_for_all") as mock_write_fits_all, \
         patch("frame.frame_pipeline._write_png_for_all") as mock_write_png_all:

        generate_frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    assert mock_bias.call_count == 2
    assert mock_dark.call_count == 2
    mock_science.assert_not_called()

    # Only bias/dark FITS are written
    mock_write_fits_all.assert_called_once()

    # PNGs are produced only for bias/dark in this configuration; no science PNGs since n_science_frames_per_channel == 0
    mock_write_png_all.assert_called_once()

