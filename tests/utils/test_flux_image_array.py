"""Tests for utils.flux_image_array."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from pathlib import Path

from utils import flux_image_array


def test_plot_flux_and_photons_windows_respects_enabled_channels(monkeypatch, tmp_path, make_star):
    """plot_flux_and_photons_windows only writes PNGs for enabled channels."""

    class _Cfg:
        run_nuv = True
        run_vis = False
        run_nir = True

    import configs.global_config as gc

    monkeypatch.setattr(gc, "_GLOBAL", _Cfg())

    wavelengths = np.linspace(1000.0, 2000.0, 128, dtype=float)
    values = np.ones_like(wavelengths)
    s = make_star(name="HD 2685")

    flux_image_array.plot_flux_and_photons_windows(
        wavelengths,
        values,
        tmp_path,
        s,
        "FluxTest",
        "Photon Flux",
        "Flux",
        perChannel=True,
        full=False,
        zoom=True,
    )

    safe_name = "HD_2685"

    def _exists(suffix: str) -> bool:
        return (Path(tmp_path) / f"{safe_name}_FluxTest_{suffix}.png").exists()

    # NUV/NIR enabled → their per-channel and zoom plots should exist
    assert _exists("NUV")
    assert _exists("NIR")
    assert _exists("NUV_zoom")
    assert _exists("NIR_zoom")

    # VIS disabled → no VIS plots
    assert not _exists("VIS")
    assert not _exists("VIS_zoom")


def test_plot_flux_and_photons_windows_full_writes_full_range_png(monkeypatch, tmp_path, make_star):
    """plot_flux_and_photons_windows with full=True writes the full-range PNG (wavelength min–max)."""
    class _Cfg:
        run_nuv = True
        run_vis = False
        run_nir = False

    import configs.global_config as gc
    monkeypatch.setattr(gc, "_GLOBAL", _Cfg())

    wavelengths = np.linspace(2000.0, 15000.0, 128, dtype=float)
    values = np.ones_like(wavelengths)
    s = make_star(name="HD 2685")

    flux_image_array.plot_flux_and_photons_windows(
        wavelengths,
        values,
        tmp_path,
        s,
        "FullTest",
        "Photon Flux",
        "Flux",
        perChannel=False,
        full=True,
        zoom=False,
    )

    full_path = Path(tmp_path) / "HD_2685_FullTest_full.png"
    assert full_path.exists()
    assert full_path.stat().st_size > 0


def test_plot_1d_for_channel_writes_expected_files(tmp_path, make_star):
    """plot_1d_for_channel writes full and zoom PNGs for a single channel."""
    wavelengths = np.linspace(1000.0, 3000.0, 256, dtype=float)
    values = np.linspace(0.0, 1.0, 256, dtype=float)
    s = make_star(name="HD 2685")

    flux_image_array.plot_1d_for_channel(
        wavelengths,
        values,
        tmp_path,
        s,
        "Counts",
        "Convolved Counts",
        "Counts s⁻¹ pixel⁻¹",
        channel_name="NUV",
        full=True,
        zoom=True,
    )

    safe_name = "HD_2685"
    full_path = Path(tmp_path) / f"{safe_name}_Counts_NUV.png"
    zoom_path = Path(tmp_path) / f"{safe_name}_Counts_NUV_zoom.png"

    assert full_path.exists()
    assert full_path.stat().st_size > 0
    assert zoom_path.exists()
    assert zoom_path.stat().st_size > 0

