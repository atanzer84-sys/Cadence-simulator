import numpy as np
import pytest

from instrument.wavelength_range import compute_extended_wavelength_range, get_required_wavelength_range


# Tests: test_empty_channels_raises
# Behavior: Empty channel list raises ValueError
def test_empty_channels_raises():
    with pytest.raises(ValueError, match="At least one channel must be provided"):
        compute_extended_wavelength_range([])


# Tests: test_single_channel_default_margin
# Behavior: Single channel uses default margin on both sides
def test_single_channel_default_margin(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        effective_area_wavelength=np.array([1000.0, 2000.0], dtype=float),
    )

    wl_min, wl_max = compute_extended_wavelength_range([ch])

    assert wl_min == 800.0
    assert wl_max == 2200.0


# Tests: test_single_channel_custom_margin
# Behavior: Single channel uses custom margin
def test_single_channel_custom_margin(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        effective_area_wavelength=np.array([500.0, 600.0], dtype=float),
    )

    wl_min, wl_max = compute_extended_wavelength_range([ch], margin_A=50.0)

    assert wl_min == 450.0
    assert wl_max == 650.0


# Tests: test_multiple_channels_takes_min_max_across_channels
# Behavior: Multiple channels use global min and max bounds
def test_multiple_channels_takes_min_max_across_channels(
    make_spectroscopy_channel,
    make_photometry_channel,
):
    ch1 = make_spectroscopy_channel(
        effective_area_wavelength=np.array([1000.0, 1500.0], dtype=float),
    )
    ch2 = make_spectroscopy_channel(
        effective_area_wavelength=np.array([2000.0, 3000.0], dtype=float),
    )
    ch3 = make_photometry_channel(
        effective_area_wavelength=np.array([500.0, 2500.0], dtype=float),
    )

    wl_min, wl_max = compute_extended_wavelength_range([ch1, ch2, ch3])

    assert wl_min == 300.0
    assert wl_max == 3200.0


# Tests: test_multiple_channels_custom_margin
# Behavior: Multiple channels use the given custom margin
def test_multiple_channels_custom_margin(make_spectroscopy_channel):
    ch1 = make_spectroscopy_channel(
        effective_area_wavelength=np.array([100.0, 200.0], dtype=float),
    )
    ch2 = make_spectroscopy_channel(
        effective_area_wavelength=np.array([150.0, 250.0], dtype=float),
    )

    wl_min, wl_max = compute_extended_wavelength_range([ch1, ch2], margin_A=10.0)

    assert wl_min == 90.0
    assert wl_max == 260.0


# Tests: test_all_none_raises
# Behavior: All missing channels raise ValueError
def test_all_none_raises():
    with pytest.raises(ValueError, match="At least one channel must be provided"):
        get_required_wavelength_range(None, None, None)


# Tests: test_nuv_only
# Behavior: NUV-only range uses default margin
def test_nuv_only(make_spectroscopy_channel):
    nuv = make_spectroscopy_channel(
        effective_area_wavelength=np.array([2000.0, 3000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(nuv, None, None)

    assert wl_min == 1800.0
    assert wl_max == 3200.0


# Tests: test_vis_only
# Behavior: VIS-only range uses default margin
def test_vis_only(make_spectroscopy_channel):
    vis = make_spectroscopy_channel(
        effective_area_wavelength=np.array([4000.0, 6000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(None, vis, None)

    assert wl_min == 3800.0
    assert wl_max == 6200.0


# Tests: test_nir_only
# Behavior: NIR-only range uses default margin
def test_nir_only(make_photometry_channel):
    nir = make_photometry_channel(
        effective_area_wavelength=np.array([10000.0, 18000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(None, None, nir)

    assert wl_min == 9800.0
    assert wl_max == 18200.0


# Tests: test_nuv_and_vis
# Behavior: NUV and VIS range spans both channels
def test_nuv_and_vis(make_spectroscopy_channel):
    nuv = make_spectroscopy_channel(
        effective_area_wavelength=np.array([2000.0, 3000.0], dtype=float),
    )
    vis = make_spectroscopy_channel(
        effective_area_wavelength=np.array([4000.0, 6000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(nuv, vis, None)

    assert wl_min == 1800.0
    assert wl_max == 6200.0


# Tests: test_nuv_and_nir
# Behavior: NUV and NIR range spans both channels
def test_nuv_and_nir(make_spectroscopy_channel, make_photometry_channel):
    nuv = make_spectroscopy_channel(
        effective_area_wavelength=np.array([2000.0, 3000.0], dtype=float),
    )
    nir = make_photometry_channel(
        effective_area_wavelength=np.array([10000.0, 18000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(nuv, None, nir)

    assert wl_min == 1800.0
    assert wl_max == 18200.0


# Tests: test_vis_and_nir
# Behavior: VIS and NIR range spans both channels
def test_vis_and_nir(make_spectroscopy_channel, make_photometry_channel):
    vis = make_spectroscopy_channel(
        effective_area_wavelength=np.array([4000.0, 6000.0], dtype=float),
    )
    nir = make_photometry_channel(
        effective_area_wavelength=np.array([10000.0, 18000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(None, vis, nir)

    assert wl_min == 3800.0
    assert wl_max == 18200.0


# Tests: test_all_three_channels
# Behavior: All channels contribute to the total range
def test_all_three_channels(make_spectroscopy_channel, make_photometry_channel):
    nuv = make_spectroscopy_channel(
        effective_area_wavelength=np.array([2000.0, 3000.0], dtype=float),
    )
    vis = make_spectroscopy_channel(
        effective_area_wavelength=np.array([4000.0, 6000.0], dtype=float),
    )
    nir = make_photometry_channel(
        effective_area_wavelength=np.array([10000.0, 18000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(nuv, vis, nir)

    assert wl_min == 1800.0
    assert wl_max == 18200.0


# Tests: test_custom_margin
# Behavior: Custom margin overrides the default margin
def test_custom_margin(make_spectroscopy_channel):
    nuv = make_spectroscopy_channel(
        effective_area_wavelength=np.array([1000.0, 2000.0], dtype=float),
    )

    wl_min, wl_max = get_required_wavelength_range(nuv, None, None, margin_A=100.0)

    assert wl_min == 900.0
    assert wl_max == 2100.0