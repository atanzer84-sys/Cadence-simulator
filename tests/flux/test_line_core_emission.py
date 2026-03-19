import numpy as np
import pytest

from utils.constants import MgII1w, MgII2w
from flux.cute_line_core_emission import apply_line_core_emission, compute_Rmg, gaussian


# Helper: generate a flat continuum flux array
def _make_flat_flux(wl_start=2700.0, wl_end=2900.0, step=0.01, level=1.0):
    wl = np.arange(wl_start, wl_end + step, step)
    flux = np.zeros((wl.size, 2), dtype=float)
    flux[:, 0] = wl
    flux[:, 1] = level
    return flux


# Helper: find nearest wavelength index
def _nearest_index(wl_array, target_wl):
    return int(np.abs(wl_array - target_wl).argmin())


# Tests: compute_Rmg
# Behavior: F/G dwarf branch uses coefficients c1=0.87, c2=5.73
def test_compute_Rmg_fg_branch():
    logR = -4.5
    expected = 10 ** (0.87 * logR + 5.73)
    got = compute_Rmg("G2V", logR)
    assert got == pytest.approx(expected, rel=0.0, abs=0.0)


# Tests: compute_Rmg
# Behavior: K dwarf branch uses coefficients c1=1.01, c2=6.00
def test_compute_Rmg_k_branch():
    logR = -4.5
    expected = 10 ** (1.01 * logR + 6.00)
    got = compute_Rmg("K0V", logR)
    assert got == pytest.approx(expected, rel=0.0, abs=0.0)


# Tests: compute_Rmg
# Behavior: M dwarf branch uses coefficients c1=1.59, c2=6.96
def test_compute_Rmg_m_branch():
    logR = -4.5
    expected = 10 ** (1.59 * logR + 6.96)
    got = compute_Rmg("M0V", logR)
    assert got == pytest.approx(expected, rel=0.0, abs=0.0)


# Tests: compute_Rmg
# Behavior: unknown spectral types return Rmg = 0
def test_compute_Rmg_unknown_returns_zero():
    got = compute_Rmg("B2V", -4.5)
    assert got == 0.0


# Tests: gaussian
# Behavior: Gaussian peaks at wl0, symmetric around wl0, and decays away from center
def test_gaussian_peaks_and_is_symmetric():
    wl0 = 2800.0
    sigma = 0.2
    scale = 3.0

    x = np.array([wl0 - sigma, wl0, wl0 + sigma], dtype=float)
    y = gaussian(x, wl0, sigma, scale=scale)

    assert y[1] == pytest.approx(scale, rel=0.0, abs=0.0)
    assert y[0] == pytest.approx(y[2], rel=0.0, abs=0.0)
    assert y[0] < y[1]


# Tests: apply_line_core_emission
# Behavior: adds emission peaks at Mg II h & k line centers, leaves far continuum unchanged
def test_apply_line_core_emission_adds_peaks_and_keeps_far_continuum():
    flux = _make_flat_flux(level=1.0)
    flux_in = flux.copy()

    sigmaMg22 = 0.257
    sigmaMg21 = 0.288
    logR = -4.5

    out = apply_line_core_emission(flux, sigmaMg22, sigmaMg21, logR, "G2V")

    wl = out[:, 0]
    i_h = _nearest_index(wl, MgII1w)
    i_k = _nearest_index(wl, MgII2w)

    # Emission increases flux at both line centers
    assert out[i_h, 1] > flux_in[i_h, 1]
    assert out[i_k, 1] > flux_in[i_k, 1]

    # Far from the lines, flux remains unchanged
    i_far = _nearest_index(wl, 2700.0)
    assert out[i_far, 1] == pytest.approx(flux_in[i_far, 1], rel=0.0, abs=1e-12)


# Tests: apply_line_core_emission
# Behavior: when Rmg = 0 (unknown spectral type), emission is zero and flux is unchanged
def test_apply_line_core_emission_noop_when_Rmg_zero():
    flux = _make_flat_flux(level=1.0)
    flux_in = flux.copy()

    sigmaMg22 = 0.257
    sigmaMg21 = 0.288

    out = apply_line_core_emission(flux, sigmaMg22, sigmaMg21, logR=-4.5, spectral_type="B2V")

    np.testing.assert_allclose(out, flux_in, rtol=0.0, atol=0.0)
