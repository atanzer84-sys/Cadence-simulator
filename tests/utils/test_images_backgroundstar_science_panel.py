"""Rudimentary tests for utils.images_backgroundstar_science_panel."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

panel = pytest.importorskip("utils.images_backgroundstar_science_panel")


def test_compute_display_range_flat_array_returns_valid_span():
    arr = np.ones((4, 4), dtype=np.float32) * 7.0
    vmin, vmax = panel._compute_display_range(arr)
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmax > vmin


def test_prepare_background_star_panel_stats_returns_three_stat_rows(make_spectroscopy_channel):
    merged = np.zeros((3, 3), dtype=np.float32)
    bg = np.zeros((3, 3), dtype=np.float32)
    ch = make_spectroscopy_channel(x_pixels=3, y_pixels=3)

    stats = panel._prepare_background_star_panel_stats(merged, bg, ch)

    assert len(stats) == 3
    assert all(len(keys) > 0 for _, keys in stats)


def test_compute_bg_mask_overlay_with_arcs_uses_arc_mask(monkeypatch, make_photometry_channel):
    img = np.zeros((7, 7), dtype=np.float32)
    ch = make_photometry_channel(x_pixels=7, y_pixels=7)
    arcs = {"s1": [(3, 3)]}

    monkeypatch.setattr(panel, "_compute_psf_r90_px", lambda _ch: 1.0)
    mask, overlay, has_bg = panel._compute_bg_mask_overlay(
        img,
        ch,
        background_star_bands=None,
        background_star_arcs=arcs,
    )

    assert has_bg is False
    assert mask[3, 3]
    assert np.isfinite(overlay[3, 3])
