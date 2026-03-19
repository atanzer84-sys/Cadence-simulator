import numpy as np
import pytest
from types import SimpleNamespace

from instrument.wavelength_range import compute_extended_wavelength_range, get_required_wavelength_range


def _channel(wl_min: float, wl_max: float):
    """Minimal channel-like object with effective_area_wavelength [wl_min, ..., wl_max]."""
    return SimpleNamespace(effective_area_wavelength=np.array([wl_min, wl_max], dtype=float))


def _spectroscopy_channel(wl_min: float, wl_max: float):
    """Minimal SpectroscopyChannel-like object."""
    return _channel(wl_min, wl_max)


def _photometry_channel(wl_min: float, wl_max: float):
    """Minimal PhotometryChannel-like object."""
    return _channel(wl_min, wl_max)


class TestComputeExtendedWavelengthRange:
    def test_empty_channels_raises(self):
        """compute_extended_wavelength_range raises ValueError when given an empty channel list."""
        with pytest.raises(ValueError, match="At least one channel must be provided"):
            compute_extended_wavelength_range([])

    def test_single_channel_default_margin(self):
        """With one channel, range is channel bounds extended by default 200 Å on each side."""
        ch = _channel(1000.0, 2000.0)
        wl_min, wl_max = compute_extended_wavelength_range([ch])
        assert wl_min == 800.0
        assert wl_max == 2200.0

    def test_single_channel_custom_margin(self):
        """With one channel and custom margin_A, range uses that margin instead of 200 Å."""
        ch = _channel(500.0, 600.0)
        wl_min, wl_max = compute_extended_wavelength_range([ch], margin_A=50.0)
        assert wl_min == 450.0
        assert wl_max == 650.0

    def test_multiple_channels_takes_min_max_across_channels(self):
        """With multiple channels, range is global min/max of first/last wavelengths plus default margin."""
        ch1 = _channel(1000.0, 1500.0)
        ch2 = _channel(2000.0, 3000.0)
        ch3 = _channel(500.0, 2500.0)
        wl_min, wl_max = compute_extended_wavelength_range([ch1, ch2, ch3])
        assert wl_min == 300.0  # 500 - 200
        assert wl_max == 3200.0  # 3000 + 200

    def test_multiple_channels_custom_margin(self):
        """With multiple channels and custom margin, range uses the given margin."""
        ch1 = _channel(100.0, 200.0)
        ch2 = _channel(150.0, 250.0)
        wl_min, wl_max = compute_extended_wavelength_range([ch1, ch2], margin_A=10.0)
        assert wl_min == 90.0
        assert wl_max == 260.0


class TestGetRequiredWavelengthRange:
    def test_all_none_raises(self):
        """get_required_wavelength_range raises ValueError when nuv, vis, and nir are all None."""
        with pytest.raises(ValueError, match="At least one channel must be provided"):
            get_required_wavelength_range(None, None, None)

    def test_nuv_only(self):
        """With only NUV channel set, range is NUV bounds plus default margin."""
        nuv = _spectroscopy_channel(2000.0, 3000.0)
        wl_min, wl_max = get_required_wavelength_range(nuv, None, None)
        assert wl_min == 1800.0
        assert wl_max == 3200.0

    def test_vis_only(self):
        """With only VIS channel set, range is VIS bounds plus default margin."""
        vis = _spectroscopy_channel(4000.0, 6000.0)
        wl_min, wl_max = get_required_wavelength_range(None, vis, None)
        assert wl_min == 3800.0
        assert wl_max == 6200.0

    def test_nir_only(self):
        """With only NIR channel set, range is NIR bounds plus default margin."""
        nir = _photometry_channel(10000.0, 18000.0)
        wl_min, wl_max = get_required_wavelength_range(None, None, nir)
        assert wl_min == 9800.0
        assert wl_max == 18200.0

    def test_nuv_and_vis(self):
        """With NUV and VIS set (NIR None), range spans both channel bounds plus default margin."""
        nuv = _spectroscopy_channel(2000.0, 3000.0)
        vis = _spectroscopy_channel(4000.0, 6000.0)
        wl_min, wl_max = get_required_wavelength_range(nuv, vis, None)
        assert wl_min == 1800.0
        assert wl_max == 6200.0

    def test_nuv_and_nir(self):
        """With NUV and NIR set (VIS None), range spans both channel bounds plus default margin."""
        nuv = _spectroscopy_channel(2000.0, 3000.0)
        nir = _photometry_channel(10000.0, 18000.0)
        wl_min, wl_max = get_required_wavelength_range(nuv, None, nir)
        assert wl_min == 1800.0
        assert wl_max == 18200.0

    def test_vis_and_nir(self):
        """With VIS and NIR set (NUV None), range spans both channel bounds plus default margin."""
        vis = _spectroscopy_channel(4000.0, 6000.0)
        nir = _photometry_channel(10000.0, 18000.0)
        wl_min, wl_max = get_required_wavelength_range(None, vis, nir)
        assert wl_min == 3800.0
        assert wl_max == 18200.0

    def test_all_three_channels(self):
        """With NUV, VIS, and NIR set, range spans all channel bounds plus default margin."""
        nuv = _spectroscopy_channel(2000.0, 3000.0)
        vis = _spectroscopy_channel(4000.0, 6000.0)
        nir = _photometry_channel(10000.0, 18000.0)
        wl_min, wl_max = get_required_wavelength_range(nuv, vis, nir)
        assert wl_min == 1800.0
        assert wl_max == 18200.0

    def test_custom_margin(self):
        """With custom margin_A, range uses that margin instead of the default 200 Å."""
        nuv = _spectroscopy_channel(1000.0, 2000.0)
        wl_min, wl_max = get_required_wavelength_range(nuv, None, None, margin_A=100.0)
        assert wl_min == 900.0
        assert wl_max == 2100.0
