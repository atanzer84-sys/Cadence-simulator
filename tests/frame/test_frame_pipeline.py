import numpy as np
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from frame.frame_pipeline import generate_Frames


class DummyCfg:
    """Minimal channel config; add attributes when the pipeline or its callees start using them."""

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
    ctx = SimpleNamespace(
        output_dir=tmp_path, timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )
    star = MagicMock()
    return counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star


def test_generate_Frames_minimal(tmp_path):
    """generate_Frames orchestrates config, generators, and batch writes."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_dark_and_bias_png = False

    # Prepare fake Frame-like objects with write_png and frame metadata
    def _make_frame():
        f = MagicMock()
        return f

    bias_nuv_frames = [_make_frame(), _make_frame()]
    bias_vis_frames = [_make_frame(), _make_frame()]
    dark_nuv_frames = [_make_frame(), _make_frame()]
    dark_vis_frames = [_make_frame(), _make_frame()]
    science_nuv_frames = [_make_frame()]
    science_vis_frames = [_make_frame()]

    with patch(
        "frame.frame_pipeline.get_global_config", return_value=mock_global_cfg
    ), patch(
        "frame.frame_pipeline.initialize_fits_header", return_value=MagicMock()
    ) as mock_init_header, patch(
        "frame.frame_pipeline.generate_bias_frames",
        side_effect=[bias_nuv_frames, bias_vis_frames],
    ) as mock_bias, patch(
        "frame.frame_pipeline.generate_dark_frames",
        side_effect=[dark_nuv_frames, dark_vis_frames],
    ) as mock_dark, patch(
        "frame.frame_pipeline.generate_science_frames",
        side_effect=[science_nuv_frames, science_vis_frames],
    ) as mock_science:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_init_header.assert_called_once_with(star, ctx.timestamp)
    assert mock_bias.call_count == 2
    assert mock_dark.call_count == 2
    assert mock_science.call_count == 2

    # Pipeline calls write_fits(ctx, index=k) per frame
    for k, f in enumerate(bias_nuv_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(bias_vis_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(dark_nuv_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(dark_vis_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    science_nuv_frames[0].write_fits.assert_called_once_with(ctx, index=0)
    science_vis_frames[0].write_fits.assert_called_once_with(ctx, index=0)

    # PNGs: one write_png per science frame with index
    science_nuv_frames[0].write_png.assert_called_once_with(ctx=ctx, star=star, index=0)
    science_vis_frames[0].write_png.assert_called_once_with(ctx=ctx, star=star, index=0)


def test_generate_Frames_skips_bias_dark_when_n_bias_and_darkframes_zero(tmp_path):
    """When n_bias_and_darkframes=0, bias/dark generation is skipped; science still runs."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 0
    mock_global_cfg.n_science_frames_per_channel = 1
    mock_global_cfg.write_dark_and_bias_png = False

    science_nuv_frames = [MagicMock()]
    science_vis_frames = [MagicMock()]

    with patch(
        "frame.frame_pipeline.get_global_config", return_value=mock_global_cfg
    ), patch(
        "frame.frame_pipeline.initialize_fits_header", return_value=MagicMock()
    ), patch(
        "frame.frame_pipeline.generate_bias_frames"
    ) as mock_bias, patch(
        "frame.frame_pipeline.generate_dark_frames"
    ) as mock_dark, patch(
        "frame.frame_pipeline.generate_science_frames",
        side_effect=[science_nuv_frames, science_vis_frames],
    ) as mock_science:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    mock_bias.assert_not_called()
    mock_dark.assert_not_called()
    assert mock_science.call_count == 2
    science_nuv_frames[0].write_fits.assert_called_once_with(ctx, index=0)
    science_vis_frames[0].write_fits.assert_called_once_with(ctx, index=0)


def test_generate_Frames_skips_science_when_n_science_frames_zero(tmp_path):
    """When n_science_frames_per_channel=0, science generation is skipped; bias/dark still run."""
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 0
    mock_global_cfg.write_dark_and_bias_png = False

    def _make_frame():
        return MagicMock()

    bias_nuv_frames = [_make_frame(), _make_frame()]
    bias_vis_frames = [_make_frame(), _make_frame()]
    dark_nuv_frames = [_make_frame(), _make_frame()]
    dark_vis_frames = [_make_frame(), _make_frame()]

    with patch(
        "frame.frame_pipeline.get_global_config", return_value=mock_global_cfg
    ), patch(
        "frame.frame_pipeline.initialize_fits_header", return_value=MagicMock()
    ), patch(
        "frame.frame_pipeline.generate_bias_frames",
        side_effect=[bias_nuv_frames, bias_vis_frames],
    ) as mock_bias, patch(
        "frame.frame_pipeline.generate_dark_frames",
        side_effect=[dark_nuv_frames, dark_vis_frames],
    ) as mock_dark, patch(
        "frame.frame_pipeline.generate_science_frames"
    ) as mock_science:

        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    assert mock_bias.call_count == 2
    assert mock_dark.call_count == 2
    mock_science.assert_not_called()
    for k, f in enumerate(bias_nuv_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(bias_vis_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(dark_nuv_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)
    for k, f in enumerate(dark_vis_frames):
        f.write_fits.assert_called_once_with(ctx, index=k)


def test_generate_Frames_multiple_frames_per_channel_each_get_unique_index(tmp_path):
    """
    With n_bias_and_darkframes=2 and n_science_frames_per_channel=2, each frame must be
    written with its own index (0, 1). Catches regressions where multiple frames would
    overwrite (e.g. all written with index=0).
    """
    counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star = _common_setup(tmp_path)

    mock_global_cfg = MagicMock()
    mock_global_cfg.n_bias_and_darkframes = 2
    mock_global_cfg.n_science_frames_per_channel = 2
    mock_global_cfg.write_dark_and_bias_png = False
    mock_global_cfg.write_science_frames_png = True

    def _make_frame():
        return MagicMock()

    n = 2
    bias_nuv_frames = [_make_frame() for _ in range(n)]
    bias_vis_frames = [_make_frame() for _ in range(n)]
    dark_nuv_frames = [_make_frame() for _ in range(n)]
    dark_vis_frames = [_make_frame() for _ in range(n)]
    science_nuv_frames = [_make_frame() for _ in range(n)]
    science_vis_frames = [_make_frame() for _ in range(n)]

    with patch(
        "frame.frame_pipeline.get_global_config", return_value=mock_global_cfg
    ), patch(
        "frame.frame_pipeline.initialize_fits_header", return_value=MagicMock()
    ), patch(
        "frame.frame_pipeline.generate_bias_frames",
        side_effect=[bias_nuv_frames, bias_vis_frames],
    ), patch(
        "frame.frame_pipeline.generate_dark_frames",
        side_effect=[dark_nuv_frames, dark_vis_frames],
    ), patch(
        "frame.frame_pipeline.generate_science_frames",
        side_effect=[science_nuv_frames, science_vis_frames],
    ):
        generate_Frames(counts_nuv, counts_vis, nuv_cfg, vis_cfg, ctx, star)

    # FITS: every frame must get write_fits(ctx, index=k) with k = 0, 1 (no overwrite)
    for frames in [
        bias_nuv_frames, bias_vis_frames,
        dark_nuv_frames, dark_vis_frames,
        science_nuv_frames, science_vis_frames,
    ]:
        assert len(frames) == 2
        for k, f in enumerate(frames):
            f.write_fits.assert_called_once_with(ctx, index=k)

    # PNG: each science frame must get write_png(..., index=k)
    for k, f in enumerate(science_nuv_frames):
        f.write_png.assert_called_once_with(ctx=ctx, star=star, index=k)
    for k, f in enumerate(science_vis_frames):
        f.write_png.assert_called_once_with(ctx=ctx, star=star, index=k)

