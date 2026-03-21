import numpy as np
import pytest
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, cut_wavelength_window_with_margin, gaussbroad


# Tests: counts_per_s_px_conv_per_channel
# Behavior: Interpolates broadened flux onto channel wavelengths and applies scale factors.
def test_single_channel_counts_identity_gaussbroad(monkeypatch, make_star, make_run_context, make_spectroscopy_channel, make_global_config):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.spectral_convolution.get_global_config", lambda: cfg)

    wavelength = np.array([100.0, 101.0, 102.0], dtype=float)
    broadened_flux = np.array([10.0, 20.0, 30.0], dtype=float)
    channel = make_spectroscopy_channel(effective_area_wavelength=np.array([100.0, 100.5, 101.0], dtype=float), effective_area=np.array([2.0, 2.0, 2.0], dtype=float), pixel_scale=0.01)

    counts = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, star=make_star(), ctx=make_run_context(output_dir="OUTDIR"))

    assert np.allclose(counts, np.array([0.20, 0.30, 0.40], dtype=float))

# Tests: counts_per_s_px_conv_per_channel
# Behavior: Processes multiple channels independently with matching array lengths to prevent broadcast errors.
def test_all_channels_counts_identity_gaussbroad(monkeypatch, make_star, make_run_context, make_spectroscopy_channel, make_global_config):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.spectral_convolution.get_global_config", lambda: cfg)

    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)
    star, ctx = make_star(), make_run_context(output_dir="OUTDIR")

    nuv = make_spectroscopy_channel(channel_name="NUV", effective_area_wavelength=np.array([100.0, 100.5], dtype=float), effective_area=np.array([2.0, 2.0], dtype=float), pixel_scale=0.01)
    vis = make_spectroscopy_channel(channel_name="VIS", effective_area_wavelength=np.array([101.0, 102.0], dtype=float), effective_area=np.array([1.0, 1.0], dtype=float), pixel_scale=0.01)

    nuv_counts = counts_per_s_px_conv_per_channel(photon_flux, wavelengths_total, nuv, star=star, ctx=ctx)
    vis_counts = counts_per_s_px_conv_per_channel(photon_flux, wavelengths_total, vis, star=star, ctx=ctx)

    assert np.allclose(nuv_counts, np.array([0.20, 0.30]))
    assert np.allclose(vis_counts, np.array([0.20, 0.30]))

# Tests: cut_wavelength_window_with_margin
# Behavior: Includes an exact upper-bound wavelength in the slice.
def test_cut_wavelength_window_with_margin_basic_slice_no_margin(make_spectroscopy_channel):
    wavelengths_total = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    channel = make_spectroscopy_channel(effective_area_wavelength=np.array([101.0, 103.0], dtype=float))

    f_cut, w_cut = cut_wavelength_window_with_margin(photon_flux, wavelengths_total, channel, margin_A=0.0)

    i0 = max(np.searchsorted(wavelengths_total, 101.0), 0)
    i1 = min(np.searchsorted(wavelengths_total, 103.0, side="right"), len(wavelengths_total))
    np.testing.assert_allclose(w_cut, wavelengths_total[i0:i1])
    np.testing.assert_allclose(f_cut, photon_flux[i0:i1])

# Tests: cut_wavelength_window_with_margin
# Behavior: Uses corrected production signature to verify index clamping at boundaries.
@pytest.mark.parametrize("bound,index", [("low", 0), ("high", -1)])
def test_cut_wavelength_window_with_margin_clamps_bounds(bound, index, make_spectroscopy_channel):
    wavelengths_total, photon_flux = np.array([100.0, 101.0, 102.0], dtype=float), np.array([10.0, 11.0, 12.0], dtype=float)
    channel = make_spectroscopy_channel(effective_area_wavelength=np.array([100.5, 101.5], dtype=float))

    f_cut, w_cut = cut_wavelength_window_with_margin(photon_flux, wavelengths_total, channel, margin_A=999.0)

    assert w_cut[index] == wavelengths_total[index]
    assert f_cut[index] == photon_flux[index]

# Tests: cut_wavelength_window_with_margin
# Behavior: Raises ValueError when the requested channel window (with margin) is outside the total wavelength range.
def test_cut_wavelength_window_no_overlap_raises_error(make_spectroscopy_channel):
    wavelengths_total = np.array([100.0, 110.0, 120.0], dtype=float)
    photon_flux = np.array([1.0, 1.0, 1.0], dtype=float)
    channel = make_spectroscopy_channel(effective_area_wavelength=np.array([500.0, 510.0], dtype=float))

    with pytest.raises(ValueError):
        cut_wavelength_window_with_margin(photon_flux, wavelengths_total, channel, margin_A=0.0)


# Tests: gaussbroad (via compute_broadened_channel_flux)
# Behavior: Returns flat mean flux when hwhm is much larger than the wavelength range.
def test_compute_broadened_channel_flux_extreme_hwhm_returns_mean(make_star, make_spectroscopy_channel):
    from instrument.spectral_convolution import compute_broadened_channel_flux
    
    wavelengths_total = np.linspace(100, 200, 100)
    photon_flux = np.ones_like(wavelengths_total) * 10.0
    photon_flux[50:] = 20.0 # Mean is 15.0
    
    # Large pixel scale forces the "hwhm > 5 * range" branch in gaussbroad
    channel = make_spectroscopy_channel(pixel_scale=1000.0, effective_area_wavelength=np.array([140.0, 160.0]))
    
    smoothed_flux, _ = compute_broadened_channel_flux(photon_flux, wavelengths_total, channel, make_star())
    
    assert np.allclose(smoothed_flux, 15.0)

# Tests: gaussbroad (via compute_broadened_channel_flux)
# Behavior: Returns original flux unchanged when hwhm is zero or negative (no broadening).
@pytest.mark.parametrize("hwhm_val", [0.0, -1.0])
def test_compute_broadened_channel_flux_nonpositive_hwhm(hwhm_val, make_star, make_spectroscopy_channel):
    from instrument.spectral_convolution import compute_broadened_channel_flux
    
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)
    
    # Manually override pixel_scale to simulate the hwhm input
    channel = make_spectroscopy_channel(pixel_scale=hwhm_val, effective_area_wavelength=np.array([100.0, 101.0, 102.0]))
    
    # This currently fails in prod due to divide-by-zero
    smoothed_flux, _ = compute_broadened_channel_flux(photon_flux, wavelengths_total, channel, make_star())
    
    assert np.array_equal(smoothed_flux, photon_flux)

# Tests: gaussbroad
# Behavior: Handles single-pixel wavelength arrays without division by zero in dispersion calculation.
def test_gaussbroad_single_pixel_input():
    wavelength = np.array([100.0])
    spectra = np.array([10.0])
    
    # This currently fails in prod due to (1-1) denominator
    result = gaussbroad(wavelength, spectra, hwhm=0.1)
    
    assert len(result) == 1
    assert result[0] == 10.0


# Tests: gaussbroad
# Behavior: Preserves a constant spectrum in the interior for positive broadening width.
def test_gaussbroad_constant_spectrum_preserved():
    wavelength = np.linspace(100.0, 110.0, 200)
    spectra = np.full_like(wavelength, 42.0)

    result = gaussbroad(wavelength, spectra, hwhm=0.8)

    assert result.shape == spectra.shape
    nhalf = int(3.3972872 * 0.8 / ((wavelength[-1] - wavelength[0]) / (len(wavelength) - 1)))
    interior = slice(nhalf + 2, -(nhalf + 2))
    np.testing.assert_allclose(result[interior], spectra[interior], rtol=0, atol=1e-10)