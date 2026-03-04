import numpy as np
import pytest
from types import SimpleNamespace

from instrument.spectral_convolution import (
    counts_per_s_px_conv_per_channel,
    cut_wavelength_window_with_margin,
)


def _cfg(test_mode=False, produce_plots=False):
    return SimpleNamespace(test_mode=test_mode, produce_plots=produce_plots)


def _ctx(output_dir="OUT"):
    """Minimal ctx with no-op test_mode and produce_plots to satisfy spectral_convolution hooks."""
    from loaders.run_waltzer_context import _NOOP, _NOOP_PLOTS

    return SimpleNamespace(output_dir=output_dir, test_mode=_NOOP, produce_plots=_NOOP_PLOTS)


def _dummy_star(name="TESTSTAR"):
    return SimpleNamespace(name=name)


def _channel(**kwargs):
    """Minimal channel-like object for spectral_convolution tests."""
    d = {
        "channel_name": "NUV",
        "wavelength": None,
        "effective_area": None,
        "pixel_scale": 0.01,
    }
    d.update(kwargs)
    return SimpleNamespace(**d)


def counts_per_s_px_conv_all_channels(
    photon_flux,
    wavelengths_total,
    nuv,
    vis,
    ctx,
    star,
):
    """
    Test helper that applies the per-channel counts conversion to both NUV and VIS,
    mirroring the legacy counts_per_s_px_conv_all_channels behavior.
    """
    nuv_counts = counts_per_s_px_conv_per_channel(
        photon_flux,
        wavelengths_total,
        nuv,
        star=star,
        ctx=ctx,
    )
    vis_counts = counts_per_s_px_conv_per_channel(
        photon_flux,
        wavelengths_total,
        vis,
        star=star,
        ctx=ctx,
    )
    return nuv_counts, vis_counts

def test_single_channel_counts_identity_gaussbroad():
    """Per-channel pipeline: interp onto channel.wavelength, then * pixel_scale * effective_area."""
    wavelength = np.array([100.0, 101.0, 102.0], dtype=float)
    broadened_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    channel = _channel(
        effective_area_wavelength=np.array([100.0, 100.5, 101.0], dtype=float),
        effective_area=np.array([2.0, 2.0, 2.0], dtype=float),
        pixel_scale=0.01,
    )

    counts = counts_per_s_px_conv_per_channel(
        broadened_flux,
        wavelength,
        channel,
        star=_dummy_star(),
        ctx=_ctx("OUTDIR"),
    )

    # interp: [10, 15, 20] -> *0.01 -> [0.10, 0.15, 0.20] -> *2 -> [0.20, 0.30, 0.40]
    expected = np.array([0.20, 0.30, 0.40], dtype=float)
    assert np.allclose(counts, expected)


def test_all_channels_counts_identity_gaussbroad(monkeypatch):
    """Same as single-channel, but for NUV and VIS using counts_per_s_px_conv_all_channels."""
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    nuv = _channel(
        channel_name="NUV",
        effective_area_wavelength=np.array([100.0, 100.5], dtype=float),
        effective_area=np.array([2.0, 2.0], dtype=float),
        pixel_scale=0.01,
    )
    vis = _channel(
        channel_name="VIS",
        effective_area_wavelength=np.array([101.0, 102.0], dtype=float),
        effective_area=np.array([1.0, 1.0], dtype=float),
        pixel_scale=0.01,
    )
    ctx = _ctx("OUTDIR")

    nuv_counts, vis_counts = counts_per_s_px_conv_all_channels(
        photon_flux, wavelengths_total, nuv, vis, ctx, _dummy_star()
    )

    # NUV: interp [10, 15] -> *0.01 -> [0.10,0.15] -> *2 -> [0.20,0.30]
    assert np.allclose(nuv_counts, np.array([0.20, 0.30]))

    # VIS: interp [20, 30] -> *0.01 -> [0.20,0.30] -> *1 -> [0.20,0.30]
    assert np.allclose(vis_counts, np.array([0.20, 0.30]))


def test_cut_wavelength_window_with_margin_basic_slice_no_margin():
    """cut_wavelength_window_with_margin returns expected slice when margin=0 and window inside bounds."""
    wavelengths_total = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

    channel = _channel(effective_area_wavelength=np.array([101.0, 103.0], dtype=float))

    f_cut, w_cut = cut_wavelength_window_with_margin(
        photon_flux,
        wavelengths_total,
        channel,
        output_dir="OUT",
        cfg=_cfg(),
        star=_dummy_star("S"),
        ctx=_ctx(),
        margin_A=0.0,
    )

    wl_min = channel.effective_area_wavelength[0]
    wl_max = channel.effective_area_wavelength[-1]
    i0 = max(np.searchsorted(wavelengths_total, wl_min), 0)
    i1 = min(np.searchsorted(wavelengths_total, wl_max), len(wavelengths_total))

    np.testing.assert_allclose(w_cut, wavelengths_total[i0:i1])
    np.testing.assert_allclose(f_cut, photon_flux[i0:i1])


@pytest.mark.parametrize("bound,index", [("low", 0), ("high", -1)])
def test_cut_wavelength_window_with_margin_clamps_bounds(bound, index):
    """Lower/upper index is clamped when margin extends beyond available wavelength range."""
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)
    channel = _channel(effective_area_wavelength=np.array([100.5, 101.5], dtype=float))

    f_cut, w_cut = cut_wavelength_window_with_margin(
        photon_flux,
        wavelengths_total,
        channel,
        output_dir="OUT",
        cfg=_cfg(),
        star=_dummy_star("S"),
        ctx=_ctx(),
        margin_A=999.0,
    )

    assert w_cut[index] == wavelengths_total[index]
    assert f_cut[index] == photon_flux[index]

