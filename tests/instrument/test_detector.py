import numpy as np
import pytest
from types import SimpleNamespace

from instrument.detector import counts_per_s_px_conv_per_channel
from instrument import detector


class _Cfg:
    def __init__(self, effective_area_file: str, x_pixels: int = 2, source_file: str = "channel_src", channel_name: str = "", spread_profile_file: str = ""):
        self.effective_area_file = effective_area_file
        self.x_pixels = x_pixels
        self.source_file = source_file
        self.channel_name = channel_name
        self.spread_profile_file = spread_profile_file


class _DummyGlobalCfg:
    test_mode = False
    produce_Plots = False


def _no_spread(_filename, _channel_name):
    """Return None spread data so tests don't call the real loader."""
    return None, None, None


def _dummy_star(name="TESTSTAR"):
    return SimpleNamespace(name=name)

def test_channel_calibration_is_frozen_and_has_expected_fields():
    # Verifies the ChannelCalibration dataclass stores wavelength, effective_area, and pixel_scale and is immutable (frozen).
    wl = np.array([1.0, 2.0])
    ea = np.array([0.1, 0.2])
    cal = detector.ChannelCalibration("Test", wavelength=wl, effective_area=ea, pixel_scale=0.5)

    assert cal.name == "Test"
    assert cal.wavelength is wl
    assert cal.effective_area is ea
    assert cal.pixel_scale == pytest.approx(0.5)

    with pytest.raises(Exception):
        cal.pixel_scale = 0.6  # frozen dataclass should reject reassignment

def test_load_channel_response_from_effective_area_calls_loader_three_times_with_cfg_filenames(monkeypatch):
    # Verifies load_channel_response_from_effective_area calls load_effective_area_file exactly three times with the effective_area_file from each cfg.
    calls = []

    def _fake_loader(filename):
        calls.append(filename)
        return np.array([1000.0, 1100.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    nuv_cal, vis_cal, ir_cal = detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert calls == ["nuv.txt", "vis.txt", "ir.txt"]
    assert isinstance(nuv_cal, detector.ChannelCalibration)
    assert isinstance(vis_cal, detector.ChannelCalibration)
    assert isinstance(ir_cal, detector.ChannelCalibration)
    assert nuv_cal.name == "NUV"
    assert vis_cal.name == "VIS"
    assert ir_cal.name == "IR"


def test_load_channel_response_from_effective_area_returns_calibrations_with_correct_values(monkeypatch):
    # Verifies the returned ChannelCalibration objects contain exactly the arrays and pixel scales returned by the loader for each channel.
    nuv_wl = np.array([1.0, 2.0, 3.0])
    nuv_ea = np.array([0.1, 0.2, 0.3])
    vis_wl = np.array([10.0, 20.0])
    vis_ea = np.array([1.1, 1.2])
    ir_wl = np.array([100.0, 200.0, 300.0, 400.0])
    ir_ea = np.array([2.1, 2.2, 2.3, 2.4])

    def _fake_loader(filename):
        if filename == "nuv.txt":
            return nuv_wl, nuv_ea, 0.01
        if filename == "vis.txt":
            return vis_wl, vis_ea, 0.02
        if filename == "ir.txt":
            return ir_wl, ir_ea, 0.03
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=len(nuv_wl))
    vis_cfg = _Cfg("vis.txt", x_pixels=len(vis_wl))
    ir_cfg = _Cfg("ir.txt", x_pixels=len(ir_wl))

    nuv_cal, vis_cal, ir_cal = detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert np.allclose(nuv_cal.wavelength, nuv_wl)
    assert np.allclose(nuv_cal.effective_area, nuv_ea)
    assert nuv_cal.pixel_scale == pytest.approx(0.01)
    assert nuv_cal.name == "NUV"

    assert np.allclose(vis_cal.wavelength, vis_wl)
    assert np.allclose(vis_cal.effective_area, vis_ea)
    assert vis_cal.pixel_scale == pytest.approx(0.02)
    assert vis_cal.name == "VIS"

    assert np.allclose(ir_cal.wavelength, ir_wl)
    assert np.allclose(ir_cal.effective_area, ir_ea)
    assert ir_cal.pixel_scale == pytest.approx(0.03)
    assert ir_cal.name == "IR"


def test_load_channel_response_from_effective_area_propagates_loader_error(monkeypatch):
    # Verifies that if load_effective_area_file raises for any channel, load_channel_response_from_effective_area propagates the same exception.
    def _fake_loader(filename):
        if filename == "vis.txt":
            raise ValueError("boom")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "boom" in str(exc.value)


def test_loader_raises_for_first_channel_propagates(monkeypatch):
    # Verifies that if the first channel loader call fails, the exception is propagated immediately.
    def _fake_loader(filename):
        raise ValueError("nuv failed")

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "nuv failed" in str(exc.value)


def test_loader_raises_for_second_channel_propagates(monkeypatch):
    # Verifies that if the second channel loader call fails, the exception is propagated.
    def _fake_loader(filename):
        if filename == "vis.txt":
            raise ValueError("vis failed")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "vis failed" in str(exc.value)


def test_loader_raises_for_third_channel_propagates(monkeypatch):
    # Verifies that if the third channel loader call fails, the exception is propagated.
    def _fake_loader(filename):
        if filename == "ir.txt":
            raise ValueError("ir failed")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "ir failed" in str(exc.value)


def test_channel_calibration_rejects_wrong_field_names():
    # Verifies that ChannelCalibration raises TypeError if unexpected keyword arguments are provided.
    with pytest.raises(TypeError):
        detector.ChannelCalibration(wl=np.array([1.0]), ea=np.array([0.1]), pixel_scale=0.01)


def test_single_channel_counts_identity_gaussbroad():
    """
    Verifies per-channel pipeline:
      interp onto cal.wavelength
      multiply by pixel_scale
      multiply by effective_area
      gaussbroad is identity when nhalf == 0

    We choose pixel_scale small enough to make nhalf==0 for this grid,
    so the expected array is exact and deterministic.
    """
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    cal = detector.ChannelCalibration(
        name="NUV",
        wavelength=np.array([100.0, 100.5, 101.0], dtype=float),
        effective_area=np.array([2.0, 2.0, 2.0], dtype=float),
        pixel_scale=0.01,
    )

    counts = detector.counts_per_s_px_conv_per_channel(
        photon_flux, wavelengths_total, cal, output_dir="OUTDIR", cfg=_DummyGlobalCfg(), star=_dummy_star()
    )

    # interp: [10, 15, 20]
    # per-pixel: *0.01 -> [0.10, 0.15, 0.20]
    # apply EA: *2 -> [0.20, 0.30, 0.40]
    expected = np.array([0.20, 0.30, 0.40], dtype=float)
    assert np.allclose(counts, expected)


def test_all_channels_counts_identity_gaussbroad():
    """
    Same as single-channel, but for 3 channels at once using the wrapper.
    We monkeypatch get_global_config so the wrapper doesn't depend on real config state.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(detector, "get_global_config", lambda: _DummyGlobalCfg())

        wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
        photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)

        nuv_cal = detector.ChannelCalibration(
            name="NUV",
            wavelength=np.array([100.0, 100.5], dtype=float),
            effective_area=np.array([2.0, 2.0], dtype=float),
            pixel_scale=0.01,
        )
        vis_cal = detector.ChannelCalibration(
            name="VIS",
            wavelength=np.array([101.0, 102.0], dtype=float),
            effective_area=np.array([1.0, 1.0], dtype=float),
            pixel_scale=0.01,
        )
        
        nuv_counts, vis_counts = detector.counts_per_s_px_conv_all_channels(
            photon_flux, wavelengths_total, nuv_cal, vis_cal, output_dir="OUTDIR", star=_dummy_star()
        )

        # NUV: interp [10, 15] -> *0.01 -> [0.10,0.15] -> *2 -> [0.20,0.30]
        assert np.allclose(nuv_counts, np.array([0.20, 0.30]))

        # VIS: interp [20, 30] -> *0.01 -> [0.20,0.30] -> *1 -> [0.20,0.30]
        assert np.allclose(vis_counts, np.array([0.20, 0.30]))

    finally:
        monkeypatch.undo()


def test_load_channel_response_from_effective_area_raises_if_nuv_length_does_not_match_x_pixels(monkeypatch):
    # Verifies that a ValueError is raised if NUV wavelength grid length does not match x_pixels.

    def _fake_loader(filename):
        return np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv_src")
    vis_cfg = _Cfg("vis.txt", x_pixels=3, source_file="vis_src")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=3, source_file="ir_src")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "NUV:" in str(exc.value)
    assert "nuv.txt" in str(exc.value)


def test_load_channel_response_from_effective_area_raises_if_vis_length_does_not_match_x_pixels(monkeypatch):
    # Verifies that a ValueError is raised if VIS wavelength grid length does not match x_pixels.

    def _fake_loader(filename):
        if filename == "vis.txt":
            return np.array([10.0, 20.0, 30.0]), np.array([1.0, 1.0, 1.0]), 0.02
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv_src")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis_src")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir_src")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "VIS:" in str(exc.value)
    assert "vis.txt" in str(exc.value)
    assert "vis_src" in str(exc.value)


def test_load_channel_response_from_effective_area_raises_if_ir_length_does_not_match_x_pixels(monkeypatch):
    # Verifies that a ValueError is raised if IR wavelength grid length does not match x_pixels.

    def _fake_loader(filename):
        if filename == "ir.txt":
            return np.array([100.0, 200.0, 300.0]), np.array([2.0, 2.0, 2.0]), 0.03
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv_src")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis_src")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir_src")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "IR:" in str(exc.value)
    assert "ir.txt" in str(exc.value)
    assert "ir_src" in str(exc.value)


def test_load_channel_response_from_effective_area_succeeds_when_lengths_match(monkeypatch):
    # Verifies that no error is raised when wavelength grid length matches x_pixels for all channels.

    def _fake_loader(filename):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _no_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv_src")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis_src")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir_src")

    nuv_cal, vis_cal, ir_cal = detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert nuv_cal.name == "NUV"
    assert vis_cal.name == "VIS"
    assert ir_cal.name == "IR"

def test_counts_per_channel_uses_cal_wavelength_grid_and_returns_same_length():
    # Verifies the function interpolates onto cal.wavelength and returns an array of identical length.

    wavelengths_total = np.array([100.0, 200.0, 300.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    cal = detector.ChannelCalibration(
        name="NUV",
        wavelength=np.array([150.0, 250.0], dtype=float),
        effective_area=np.array([1.0, 1.0], dtype=float),
        pixel_scale=0.01,   # small so gaussbroad is identity (nhalf=0)
    )

    out = detector.counts_per_s_px_conv_per_channel(
        photon_flux, wavelengths_total, cal, output_dir="OUTDIR", cfg=_DummyGlobalCfg(), star=_dummy_star())


    # interp at 150 -> 15, at 250 -> 25
    expected = np.array([15.0, 25.0], dtype=float) * 0.01 * 1.0

    assert out.shape == cal.wavelength.shape
    assert np.allclose(out, expected)


def test_cut_wavelength_window_with_margin_basic_slice_no_margin(monkeypatch):
    # Verifies that the function returns the exact expected flux and wavelength slice when margin is zero and window is fully inside bounds.
    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    wavelengths_total = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

    cal = SimpleNamespace(
        name="NUV",
        wavelength=np.array([101.0, 103.0], dtype=float),
    )

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, cal, output_dir="OUT", cfg=cfg, star=star, margin_A=0.0
    )

    wl_min = cal.wavelength[0]
    wl_max = cal.wavelength[-1]
    i0 = max(np.searchsorted(wavelengths_total, wl_min), 0)
    i1 = min(np.searchsorted(wavelengths_total, wl_max), len(wavelengths_total))

    np.testing.assert_allclose(w_cut, wavelengths_total[i0:i1])
    np.testing.assert_allclose(f_cut, photon_flux[i0:i1])


def test_cut_wavelength_window_with_margin_clamps_low_bound(monkeypatch):
    # Verifies that the lower index is clamped to the start of the array when the margin extends below the available wavelength range.
    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)

    cal = SimpleNamespace(name="NUV", wavelength=np.array([100.5, 101.5], dtype=float))

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, cal, output_dir="OUT", cfg=cfg, star=star, margin_A=999.0
    )

    assert w_cut[0] == wavelengths_total[0]
    assert f_cut[0] == photon_flux[0]


def test_cut_wavelength_window_with_margin_clamps_high_bound(monkeypatch):
    # Verifies that the upper index is clamped to the end of the array when the margin extends beyond the available wavelength range.
    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)

    cal = SimpleNamespace(name="NUV", wavelength=np.array([100.5, 101.5], dtype=float))

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, cal, output_dir="OUT", cfg=cfg, star=star, margin_A=999.0
    )

    assert w_cut[-1] == wavelengths_total[-1]
    assert f_cut[-1] == photon_flux[-1]


def test_cut_wavelength_window_with_margin_calls_dump_with_x_then_y(monkeypatch):
    # Verifies that in test_mode the function calls dump_1d_array with wavelength as x and flux as y.
    calls = []
    cfg = SimpleNamespace(test_mode=True, produce_Plots=False)
    star = SimpleNamespace(name="S")

    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)
    cal = SimpleNamespace(name="NUV", wavelength=np.array([100.0, 102.0], dtype=float))

    def _fake_dump(x, y, output_dir, star_name, tag, full=True, zoom=False, channel_name=None, **kwargs):
        calls.append((x.copy(), y.copy(), output_dir, star_name, tag, full, zoom, channel_name))

    monkeypatch.setattr(detector, "dump_1d_for_channel", _fake_dump)

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, cal, output_dir="OUT", cfg=cfg, star=star, margin_A=0.0
    )

    assert len(calls) == 1
    x, y, outdir, star_name, tag, full, zoom, channel_name = calls[0]
    np.testing.assert_allclose(x, w_cut)
    np.testing.assert_allclose(y, f_cut)
    assert outdir == "OUT"
    assert star_name == "S"
    assert tag == "Detector_1_cut_wavelength_window"
    assert channel_name == "NUV"
    assert full is True
    assert zoom is True


def test_compute_broadened_channel_flux_calls_cut_and_gaussbroad_in_order(monkeypatch):
    # Verifies that compute_broadened_channel_flux passes flux and wavelength in correct order to gaussbroad and returns the expected pair.
    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")
    cal = SimpleNamespace(name="NUV", pixel_scale=0.25)

    flux_cut = np.array([1.0, 2.0, 3.0], dtype=float)
    wl_cut = np.array([10.0, 11.0, 12.0], dtype=float)
    smoothed = np.array([9.0, 8.0, 7.0], dtype=float)

    def _fake_cut(photon_flux_at_earth, wavelengths_total, cal_arg, output_dir, cfg_arg, star_arg, margin_A=100.0):
        return flux_cut, wl_cut

    def _fake_gaussbroad(w, s, hwhm):
        np.testing.assert_allclose(w, wl_cut)
        np.testing.assert_allclose(s, flux_cut)
        assert hwhm == pytest.approx(cal.pixel_scale)
        return smoothed

    monkeypatch.setattr(detector, "cut_wavelength_window_with_margin", _fake_cut)
    monkeypatch.setattr(detector, "gaussbroad", _fake_gaussbroad)

    out_flux, out_wl = detector.compute_broadened_channel_flux(
        photon_flux_at_earth=np.array([0.0]),
        wavelengths_total=np.array([0.0]),
        channel=cal,
        output_dir="OUT",
        cfg=cfg,
        star=star,
    )

    np.testing.assert_allclose(out_flux, smoothed)
    np.testing.assert_allclose(out_wl, wl_cut)


def test_counts_per_s_px_conv_all_channels_calls_broaden_then_convert_for_each_channel(monkeypatch):
    # Verifies that the wrapper calls broaden and conversion once per channel and returns results in the correct order.
    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    nuv_cal = SimpleNamespace(name="NUV")
    vis_cal = SimpleNamespace(name="VIS")

    call_sequence = []

    def _fake_get_global_config():
        return cfg

    def _fake_broaden(photon_flux_at_earth, wavelengths_total, cal, output_dir, cfg_arg, star_arg):
        call_sequence.append(("broaden", cal.name))
        return np.array([1.0]), np.array([2.0])

    def _fake_conv(broadened_photon_flux, wavelength, cal, output_dir, cfg_arg, star_arg):
        call_sequence.append(("convert", cal.name))
        return np.array([cal.name], dtype=object)

    monkeypatch.setattr(detector, "get_global_config", _fake_get_global_config)
    monkeypatch.setattr(detector, "compute_broadened_channel_flux", _fake_broaden)
    monkeypatch.setattr(detector, "counts_per_s_px_conv_per_channel", _fake_conv)

    out_nuv, out_vis = detector.counts_per_s_px_conv_all_channels(
        photon_flux_at_earth=np.array([0.0]),
        wavelengths_total=np.array([0.0]),
        nuv=nuv_cal,
        vis=vis_cal,
        output_dir="OUT",
        star=star,
    )

    assert call_sequence == [
        ("broaden", "NUV"),
        ("broaden", "VIS"),
        ("convert", "NUV"),
        ("convert", "VIS"),
    ]

    assert out_nuv.tolist() == ["NUV"]
    assert out_vis.tolist() == ["VIS"]




def test_counts_per_channel_identity_scaling():
    # Verifies that with identical wavelength grids the function reduces to simple pixel_scale and effective_area scaling.
    wavelength = np.array([1.0, 2.0, 3.0, 4.0])
    broadened_flux = np.array([10.0, 20.0, 30.0, 40.0])

    cal = SimpleNamespace(
        name="TEST",
        wavelength=wavelength.copy(),
        pixel_scale=2.0,
        effective_area=np.array([5.0, 5.0, 5.0, 5.0]),
    )

    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    out = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, cal, "OUT", cfg, star
    )

    expected = broadened_flux * 2.0 * 5.0
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_interpolation_linear():
    # Verifies that linear interpolation is applied correctly before scaling.
    wavelength = np.array([0.0, 1.0, 2.0, 3.0])
    broadened_flux = np.array([0.0, 10.0, 20.0, 30.0])

    cal = SimpleNamespace(
        name="TEST",
        wavelength=np.array([0.5, 1.5, 2.5]),
        pixel_scale=1.0,
        effective_area=np.ones(3),
    )

    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    out = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, cal, "OUT", cfg, star
    )

    expected = np.array([5.0, 15.0, 25.0])
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_interpolation_clamping():
    # Verifies that interpolation outside the wavelength range clamps to boundary flux values.
    wavelength = np.array([0.0, 1.0, 2.0, 3.0])
    broadened_flux = np.array([0.0, 10.0, 20.0, 30.0])

    cal = SimpleNamespace(
        name="TEST",
        wavelength=np.array([-1.0, 0.0, 3.0, 5.0]),
        pixel_scale=1.0,
        effective_area=np.ones(4),
    )

    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    out = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, cal, "OUT", cfg, star
    )

    expected = np.array([0.0, 0.0, 30.0, 30.0])
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_output_shape_matches_calibration():
    # Verifies that the output array shape always matches the calibration wavelength grid.
    wavelength = np.array([0.0, 1.0, 2.0])
    broadened_flux = np.array([10.0, 20.0, 30.0])

    cal = SimpleNamespace(
        name="TEST",
        wavelength=np.array([0.5, 1.5]),
        pixel_scale=1.0,
        effective_area=np.ones(2),
    )

    cfg = SimpleNamespace(test_mode=False, produce_Plots=False)
    star = SimpleNamespace(name="S")

    out = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, cal, "OUT", cfg, star
    )

    assert out.shape == cal.wavelength.shape


def test_counts_per_channel_calls_dump_in_test_mode(monkeypatch):
    # Verifies that dump_1d_for_channel is called with calibration wavelength and output values when test_mode is enabled.
    calls = []

    def fake_dump(x, y, output_dir, star_name, tag, full=True, zoom=True, channel_name=None, **kwargs):
        calls.append((x.copy(), y.copy(), tag, channel_name, full, zoom))

    wavelength = np.array([1.0, 2.0, 3.0])
    broadened_flux = np.array([10.0, 20.0, 30.0])

    cal = SimpleNamespace(
        name="TEST",
        wavelength=wavelength.copy(),
        pixel_scale=1.0,
        effective_area=np.ones(3),
    )

    cfg = SimpleNamespace(test_mode=True, produce_Plots=False)
    star = SimpleNamespace(name="S")

    from instrument import detector
    monkeypatch.setattr(detector, "dump_1d_for_channel", fake_dump)

    out = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, cal, "OUT", cfg, star)

    assert len(calls) == 1
    x, y, tag, channel_name, full, zoom = calls[0]
    np.testing.assert_allclose(x, cal.wavelength)
    np.testing.assert_allclose(y, out)
    assert tag == f"Detector_2_counts_s_px_convolved_{cal.name}"
    assert channel_name == cal.name
    assert full is True
    assert zoom is True



def test_load_channel_response_from_effective_area_sets_spread_fields_from_loader(monkeypatch):
    # Verifies spread loader outputs are stored on ChannelCalibration objects for each channel.
    def _fake_ea_loader(filename):
        if filename == "nuv.txt":
            return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01
        if filename == "vis.txt":
            return np.array([1.0, 2.0]), np.array([0.3, 0.4]), 0.02
        if filename == "ir.txt":
            return np.array([1.0, 2.0]), np.array([0.5, 0.6]), 0.03
        raise AssertionError("unexpected")

    def _fake_spread_loader(filename, channel_name):
        if channel_name == "NUV":
            return np.array([0.0, 1.0]), np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([10.0, 20.0])
        if channel_name == "VIS":
            return np.array([0.0]), np.array([[0.9, 1.1]]), np.array([30.0, 40.0])
        if channel_name == "IR":
            return None, None, None
        raise AssertionError("unexpected")

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_ea_loader)
    monkeypatch.setattr(detector, "load_spread_profile_file", _fake_spread_loader)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, channel_name="NUV", spread_profile_file="nuv_spread.txt")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, channel_name="VIS", spread_profile_file="vis_spread.txt")
    ir_cfg = _Cfg("ir.txt",  x_pixels=2, channel_name="IR",  spread_profile_file="")

    nuv_cal, vis_cal, ir_cal = detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert np.allclose(nuv_cal.spread_y_positions, np.array([0.0, 1.0]))
    assert nuv_cal.spread_y_weights.shape == (2, 2)
    assert np.allclose(nuv_cal.spread_y_wavelengths, np.array([10.0, 20.0]))

    assert np.allclose(vis_cal.spread_y_positions, np.array([0.0]))
    assert vis_cal.spread_y_weights.shape == (1, 2)
    assert np.allclose(vis_cal.spread_y_wavelengths, np.array([30.0, 40.0]))

    assert ir_cal.spread_y_positions is None
    assert ir_cal.spread_y_weights is None
    assert ir_cal.spread_y_wavelengths is None

def test_load_channel_response_from_effective_area_calls_spread_loader_with_empty_filename_and_sets_none(monkeypatch):
    # Verifies empty spread_profile_file is passed through and results in None spread fields.
    monkeypatch.setattr(detector, "load_effective_area_file", lambda filename: (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01))

    calls = []

    def _fake_spread(filename, channel_name):
        calls.append((filename, channel_name))
        return None, None, None

    monkeypatch.setattr(detector, "load_spread_profile_file", _fake_spread)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, channel_name="NUV", spread_profile_file="")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, channel_name="VIS", spread_profile_file="")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, channel_name="IR",  spread_profile_file="")

    nuv_cal, _, _ = detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert calls == [("", "NUV"), ("", "VIS"), ("", "IR")]
    assert nuv_cal.spread_y_positions is None
    assert nuv_cal.spread_y_weights is None
    assert nuv_cal.spread_y_wavelengths is None

