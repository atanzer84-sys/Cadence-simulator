import numpy as np
import pytest

from instrument import calibration


class _Cfg:
    def __init__(self, effective_area_file: str):
        self.effective_area_file = effective_area_file


def test_channel_calibration_is_frozen_and_has_expected_fields():
    # Verifies the ChannelCalibration dataclass stores wavelength, effective_area, and pixel_scale and is immutable (frozen).
    wl = np.array([1.0, 2.0])
    ea = np.array([0.1, 0.2])
    cal = calibration.ChannelCalibration(wavelength=wl, effective_area=ea, pixel_scale=0.5)

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

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    nuv_cal, vis_cal, ir_cal = calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert calls == ["nuv.txt", "vis.txt", "ir.txt"]
    assert isinstance(nuv_cal, calibration.ChannelCalibration)
    assert isinstance(vis_cal, calibration.ChannelCalibration)
    assert isinstance(ir_cal, calibration.ChannelCalibration)


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

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    nuv_cal, vis_cal, ir_cal = calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert np.allclose(nuv_cal.wavelength, nuv_wl)
    assert np.allclose(nuv_cal.effective_area, nuv_ea)
    assert nuv_cal.pixel_scale == pytest.approx(0.01)

    assert np.allclose(vis_cal.wavelength, vis_wl)
    assert np.allclose(vis_cal.effective_area, vis_ea)
    assert vis_cal.pixel_scale == pytest.approx(0.02)

    assert np.allclose(ir_cal.wavelength, ir_wl)
    assert np.allclose(ir_cal.effective_area, ir_ea)
    assert ir_cal.pixel_scale == pytest.approx(0.03)


def test_load_instrument_calibration_propagates_loader_error(monkeypatch):
    # Verifies that if load_effective_area_file raises for any channel, load_instrument_calibration propagates the same exception.
    def _fake_loader(filename):
        if filename == "vis.txt":
            raise ValueError("boom")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert "boom" in str(exc.value)


def test_loader_raises_for_first_channel_propagates(monkeypatch):
    # Verifies that if the first channel loader call fails, the exception is propagated immediately.
    def _fake_loader(filename):
        raise ValueError("nuv failed")

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert "nuv failed" in str(exc.value)


def test_loader_raises_for_second_channel_propagates(monkeypatch):
    # Verifies that if the second channel loader call fails, the exception is propagated.
    def _fake_loader(filename):
        if filename == "vis.txt":
            raise ValueError("vis failed")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert "vis failed" in str(exc.value)


def test_loader_raises_for_third_channel_propagates(monkeypatch):
    # Verifies that if the third channel loader call fails, the exception is propagated.
    def _fake_loader(filename):
        if filename == "ir.txt":
            raise ValueError("ir failed")
        return np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.01

    monkeypatch.setattr(calibration, "load_effective_area_file", _fake_loader)

    nuv_cfg = _Cfg("nuv.txt")
    vis_cfg = _Cfg("vis.txt")
    ir_cfg = _Cfg("ir.txt")

    with pytest.raises(ValueError) as exc:
        calibration.load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg)

    assert "ir failed" in str(exc.value)


def test_channel_calibration_rejects_wrong_field_names():
    # Verifies that ChannelCalibration raises TypeError if unexpected keyword arguments are provided.
    with pytest.raises(TypeError):
        calibration.ChannelCalibration(wl=np.array([1.0]), ea=np.array([0.1]), pixel_scale=0.01)
