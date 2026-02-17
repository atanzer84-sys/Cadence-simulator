import numpy as np
import pytest
from types import SimpleNamespace

from instrument import detector

class _Cfg:
    def __init__(self, effective_area_file: str, x_pixels: int = 2, source_file: str = "channel.cfg"):
        self.effective_area_file = effective_area_file
        self.x_pixels = x_pixels
        self.source_file = source_file


class _DummyGlobalCfg:
    test_mode = False
    produce_Plots = False

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

    counts = detector.counts_per_s_px_conv_all_channels_per_channel(
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
        ir_cal = detector.ChannelCalibration(
            name="IR",
            wavelength=np.array([100.0, 102.0], dtype=float),
            effective_area=np.array([3.0, 3.0], dtype=float),
            pixel_scale=0.01,
        )

        nuv_counts, vis_counts, ir_counts = detector.counts_per_s_px_conv_all_channels(
            photon_flux, wavelengths_total, nuv_cal, vis_cal, ir_cal, output_dir="OUTDIR", star=_dummy_star()
        )

        # NUV: interp [10, 15] -> *0.01 -> [0.10,0.15] -> *2 -> [0.20,0.30]
        assert np.allclose(nuv_counts, np.array([0.20, 0.30]))

        # VIS: interp [20, 30] -> *0.01 -> [0.20,0.30] -> *1 -> [0.20,0.30]
        assert np.allclose(vis_counts, np.array([0.20, 0.30]))

        # IR:  interp [10, 30] -> *0.01 -> [0.10,0.30] -> *3 -> [0.30,0.90]
        assert np.allclose(ir_counts, np.array([0.30, 0.90]))
    finally:
        monkeypatch.undo()


def test_load_channel_response_from_effective_area_raises_if_nuv_length_does_not_match_x_pixels(monkeypatch):
    # Verifies that a ValueError is raised if NUV wavelength grid length does not match x_pixels.

    def _fake_loader(filename):
        return np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv.cfg")
    vis_cfg = _Cfg("vis.txt", x_pixels=3, source_file="vis.cfg")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=3, source_file="ir.cfg")

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

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv.cfg")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis.cfg")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir.cfg")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "VIS:" in str(exc.value)
    assert "vis.txt" in str(exc.value)
    assert "vis.cfg" in str(exc.value)


def test_load_channel_response_from_effective_area_raises_if_ir_length_does_not_match_x_pixels(monkeypatch):
    # Verifies that a ValueError is raised if IR wavelength grid length does not match x_pixels.

    def _fake_loader(filename):
        if filename == "ir.txt":
            return np.array([100.0, 200.0, 300.0]), np.array([2.0, 2.0, 2.0]), 0.03
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv.cfg")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis.cfg")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir.cfg")

    with pytest.raises(ValueError) as exc:
        detector.load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)

    assert "IR:" in str(exc.value)
    assert "ir.txt" in str(exc.value)
    assert "ir.cfg" in str(exc.value)


def test_load_channel_response_from_effective_area_succeeds_when_lengths_match(monkeypatch):
    # Verifies that no error is raised when wavelength grid length matches x_pixels for all channels.

    def _fake_loader(filename):
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt", x_pixels=2, source_file="nuv.cfg")
    vis_cfg = _Cfg("vis.txt", x_pixels=2, source_file="vis.cfg")
    ir_cfg  = _Cfg("ir.txt",  x_pixels=2, source_file="ir.cfg")

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

    out = detector.counts_per_s_px_conv_all_channels_per_channel(
        photon_flux, wavelengths_total, cal, output_dir="OUTDIR", cfg=_DummyGlobalCfg(), star=_dummy_star())


    # interp at 150 -> 15, at 250 -> 25
    expected = np.array([15.0, 25.0], dtype=float) * 0.01 * 1.0

    assert out.shape == cal.wavelength.shape
    assert np.allclose(out, expected)

def test_counts_per_channel_interpolates_scales_and_applies_effective_area(monkeypatch):

    monkeypatch.setattr(detector, "gaussbroad", lambda wl, y, pixel_scale: y)

    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux_at_earth = np.array([10.0, 20.0, 30.0], dtype=float)

    cal = SimpleNamespace(
        name="NUV",
        wavelength=np.array([100.0, 100.5, 101.0], dtype=float),
        effective_area=np.array([2.0, 2.0, 2.0], dtype=float),
        pixel_scale=0.5,
    )

    out = detector.counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth, wavelengths_total, cal, output_dir=None, cfg=_DummyGlobalCfg(), star=_dummy_star())

    flux_on_pixel = np.interp(cal.wavelength, wavelengths_total, photon_flux_at_earth)
    expected = flux_on_pixel * cal.pixel_scale * cal.effective_area

    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)

def test_counts_per_channel_calls_gaussbroad_with_pixel_grid_and_pixel_scale(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from instrument import detector

    calls = []

    def _fake_gaussbroad(wl, y, pixel_scale):
        calls.append((wl.copy(), y.copy(), float(pixel_scale)))
        return y

    monkeypatch.setattr(detector, "gaussbroad", _fake_gaussbroad)

    wavelengths_total = np.array([1.0, 2.0], dtype=float)
    photon_flux_at_earth = np.array([10.0, 20.0], dtype=float)

    cal = SimpleNamespace(
        name="VIS",
        wavelength=np.array([1.5], dtype=float),
        effective_area=np.array([3.0], dtype=float),
        pixel_scale=0.2,
    )

    _ = detector.counts_per_s_px_conv_all_channels_per_channel(
        photon_flux_at_earth, wavelengths_total, cal, output_dir=None, cfg=_DummyGlobalCfg(), star=_dummy_star()
    )

    assert len(calls) == 1
    wl_arg, y_arg, ps_arg = calls[0]

    assert np.allclose(wl_arg, cal.wavelength)
    assert ps_arg == cal.pixel_scale

    # expected y passed to gaussbroad: interp onto cal.wavelength, then *pixel_scale, then *effective_area
    expected_flux_on_pixel = np.interp(cal.wavelength, wavelengths_total, photon_flux_at_earth)
    expected_y = expected_flux_on_pixel * cal.pixel_scale * cal.effective_area
    assert np.allclose(y_arg, expected_y)
