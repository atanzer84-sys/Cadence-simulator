import numpy as np
import pytest

from instrument import detector

class _Cfg:
    def __init__(self, effective_area_file: str, x_pixels: int = 2):
        self.effective_area_file = effective_area_file
        self.x_pixels = x_pixels


class _DummyGlobalCfg:
    # detector.counts_per_s_px_conv_all_channels_per_channel checks only this
    test_mode = False

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



def test_load_instrument_calibration_calls_loader_three_times_with_cfg_filenames(monkeypatch):
    # Verifies load_instrument_calibration calls load_effective_area_file exactly three times with the effective_area_file from each cfg.
    calls = []

    def _fake_loader(filename):
        calls.append(filename)
        return np.array([1000.0, 1100.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    nuv_cal, vis_cal, ir_cal = detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

    assert calls == ["nuv.txt", "vis.txt", "ir.txt"]
    assert isinstance(nuv_cal, detector.ChannelCalibration)
    assert isinstance(vis_cal, detector.ChannelCalibration)
    assert isinstance(ir_cal, detector.ChannelCalibration)
    assert nuv_cal.name == "NUV"
    assert vis_cal.name == "VIS"
    assert ir_cal.name == "IR"


def test_load_instrument_calibration_returns_calibrations_with_correct_values(monkeypatch):
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


    nuv_cal, vis_cal, ir_cal = detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

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


def test_load_instrument_calibration_propagates_loader_error(monkeypatch):
    # Verifies that if load_effective_area_file raises for any channel, load_instrument_calibration propagates the same exception.
    def _fake_loader(filename):
        if filename == "vis.txt":
            raise ValueError("boom")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(detector, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

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
        detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

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
        detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

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
        detector.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg, out="OUTDIR")

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
        photon_flux, wavelengths_total, cal, output_dir="OUTDIR", cfg=_DummyGlobalCfg()
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
            photon_flux, wavelengths_total, nuv_cal, vis_cal, ir_cal, output_dir="OUTDIR"
        )

        # NUV: interp [10, 15] -> *0.01 -> [0.10,0.15] -> *2 -> [0.20,0.30]
        assert np.allclose(nuv_counts, np.array([0.20, 0.30]))

        # VIS: interp [20, 30] -> *0.01 -> [0.20,0.30] -> *1 -> [0.20,0.30]
        assert np.allclose(vis_counts, np.array([0.20, 0.30]))

        # IR:  interp [10, 30] -> *0.01 -> [0.10,0.30] -> *3 -> [0.30,0.90]
        assert np.allclose(ir_counts, np.array([0.30, 0.90]))
    finally:
        monkeypatch.undo()
