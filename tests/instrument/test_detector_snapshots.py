
import numpy as np
from pathlib import Path
from types import SimpleNamespace 
from instrument.detector import ChannelCalibration, counts_per_s_px_conv_all_channels_per_channel

class _Cfg:
    test_mode = False
    produce_Plots = False

SNAPSHOT_DIR = Path("tests/instrument/snapshots")

star = SimpleNamespace(name="star_name")

def run_detector_snapshot(star_name, channel):
    inp = np.loadtxt(SNAPSHOT_DIR / f"{star_name}_input_totalgrid_{channel}.txt", dtype=np.float64)
    wavelengths_total = inp[:, 0]
    photon_flux_at_earth = inp[:, 1]

    cal_wavelength = np.loadtxt(SNAPSHOT_DIR / f"{star_name}_cal_wavelength_{channel}.txt", dtype=np.float64)
    cal_effective_area = np.loadtxt(SNAPSHOT_DIR / f"{star_name}_cal_effective_area_{channel}.txt", dtype=np.float64)
    pixel_scale = float(np.loadtxt(SNAPSHOT_DIR / f"{star_name}_cal_pixel_scale_{channel}.txt", dtype=np.float64))


    expected = np.loadtxt(SNAPSHOT_DIR / f"{star_name}_expected_counts_per_s_per_pixel_convolved_{channel}.txt", dtype=np.float64)
    expected_counts = expected[:, 1]

    cal = ChannelCalibration(channel, cal_wavelength, cal_effective_area, pixel_scale)

    out = counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth, wavelengths_total, cal, SNAPSHOT_DIR, _Cfg(), star)

    assert out.shape == expected_counts.shape
    np.testing.assert_allclose(out, expected_counts, rtol=1e-10, atol=0.0)


def test_HD2685_detector_counts_NUV():
    run_detector_snapshot("HD 2685", "NUV")


def test_HD2685_detector_counts_VIS():
    run_detector_snapshot("HD 2685", "VIS")


def test_HD2685_detector_counts_IR():
    run_detector_snapshot("HD 2685", "IR")


def test_WASP69_detector_counts_NUV():
    run_detector_snapshot("WASP-69", "NUV")


def test_WASP69_detector_counts_VIS():
    run_detector_snapshot("WASP-69", "VIS")


def test_WASP69_detector_counts_IR():
    run_detector_snapshot("WASP-69", "IR")
