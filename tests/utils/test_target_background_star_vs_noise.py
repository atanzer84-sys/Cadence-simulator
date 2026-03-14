"""Tests for utils.target_background_star_vs_noise."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path

from tests.helpers.channel_factory import channel
from tests.helpers.star_factory import star
from tests.helpers.run_context_factory import run_context
from utils import target_background_star_vs_noise


def test_plot_star_counts_vs_noise_spectroscopy_writes_png(tmp_path):
    """plot_star_counts_vs_noise_spectroscopy writes a PNG to ctx.output_dir."""
    wavelength = np.linspace(4000.0, 8000.0, 50, dtype=float)
    counts_s_px = np.ones(50, dtype=float) * 0.1
    ch = channel(channel_name="VIS", exposure_s=60.0)
    ctx = run_context(tmp_path)

    target_background_star_vs_noise.plot_star_counts_vs_noise_spectroscopy(
        wavelength, counts_s_px, ch, ctx, None
    )

    out = Path(tmp_path) / "HD_2685_target_star_counts_vs_noise_VIS.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_star_counts_vs_noise_spectroscopy_with_star_writes_png(tmp_path):
    """plot_star_counts_vs_noise_spectroscopy accepts star and still writes PNG."""
    wavelength = np.linspace(4000.0, 8000.0, 50, dtype=float)
    counts_s_px = np.ones(50, dtype=float) * 0.1
    ch = channel(channel_name="NUV")
    ctx = run_context(tmp_path)
    s = star(name="KELT-20", gaia_magnitude=7.6, effective_temperature=6000.0, distance_pc=100.0)

    target_background_star_vs_noise.plot_star_counts_vs_noise_spectroscopy(
        wavelength, counts_s_px, ch, ctx, s
    )

    out = Path(tmp_path) / "HD_2685_target_star_counts_vs_noise_NUV.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_star_counts_vs_noise_photometry_writes_png(tmp_path):
    """plot_star_counts_vs_noise_photometry writes a PNG to ctx.output_dir."""
    rate_image_e_s = np.ones((10, 10), dtype=np.float32) * 0.5
    ch = channel(channel_name="NIR", exposure_s=30.0)
    ctx = run_context(tmp_path)

    target_background_star_vs_noise.plot_star_counts_vs_noise_photometry(
        rate_image_e_s, ch, ctx, None
    )

    out = Path(tmp_path) / "HD_2685_target_star_counts_vs_noise_NIR.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_star_counts_vs_noise_photometry_with_star_writes_png(tmp_path):
    """plot_star_counts_vs_noise_photometry accepts star and still writes PNG."""
    rate_image_e_s = np.ones((8, 8), dtype=np.float32) * 1.0
    ch = channel(channel_name="NIR")
    ctx = run_context(tmp_path)
    s = star(name="WASP-99", gaia_magnitude=9.2, effective_temperature=5800.0, distance_pc=150.0)

    target_background_star_vs_noise.plot_star_counts_vs_noise_photometry(
        rate_image_e_s, ch, ctx, s
    )

    out = Path(tmp_path) / "HD_2685_target_star_counts_vs_noise_NIR.png"
    assert out.exists()
    assert out.stat().st_size > 0
