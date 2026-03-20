"""Tests for utils.debug_dumps.

Coverage in this file focuses on:
- dump_masked_1d
- dump_3d_array
- dump_1d_for_channel
"""
import numpy as np
import pytest

from utils.debug_dumps import dump_1d_for_channel, dump_3d_array, dump_masked_1d


def test_dump_masked_1d_writes_file(tmp_path):
    """dump_masked_1d writes wavelength and array columns within wmin/wmax."""
    wave = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    array = np.array([1.0, 2.0, 3.0, 4.0])
    dump_masked_1d(wave, array, tmp_path, "test.txt", 1050.0, 1250.0)

    out = tmp_path / "test.txt"
    assert out.exists()
    data = np.loadtxt(out)
    assert data.shape == (2, 2)  # 1100,1200 within range
    assert np.allclose(data[:, 0], [1100.0, 1200.0])
    assert np.allclose(data[:, 1], [2.0, 3.0])


def test_dump_masked_1d_empty_mask_writes_nothing(tmp_path):
    """dump_masked_1d with no data in range returns without writing."""
    wave = np.array([1000.0, 1100.0])
    array = np.array([1.0, 2.0])
    dump_masked_1d(wave, array, tmp_path, "empty.txt", 2000.0, 3000.0)

    assert not (tmp_path / "empty.txt").exists()


def test_dump_1d_for_channel_shape_mismatch_raises():
    """dump_1d_for_channel raises when wave and array shapes differ."""
    wave = np.array([1.0, 2.0])
    array = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="shape mismatch"):
        dump_1d_for_channel(wave, array, "/tmp", "star", "tag", "NUV")

def test_dump_3d_array_per_channel_writes_nuv_file(monkeypatch, tmp_path):
    """dump_3d_array with perChannel=True writes NUV file when run_nuv is True (config mocked)."""
    from utils.constants import debug_wavelength_range_nuv
    wmin, wmax = float(debug_wavelength_range_nuv[0]), float(debug_wavelength_range_nuv[1])
    mid = (wmin + wmax) / 2
    array = np.array([[wmin - 500, 0.0], [mid, 1.0], [wmax - 1, 2.0], [wmax + 500, 3.0]])
    cfg = type("Cfg", (), {"run_nuv": True, "run_vis": False, "run_nir": False})()
    monkeypatch.setattr("utils.debug_dumps.get_global_config", lambda: cfg)

    dump_3d_array(array, tmp_path, "star", "tag", perChannel=True)

    out = tmp_path / "star_tag_NUV.txt"
    assert out.exists()
    data = np.loadtxt(out)
    assert data.shape[0] == 2
    assert np.allclose(data[:, 0], [mid, wmax - 1])
    assert np.allclose(data[:, 1], [1.0, 2.0])


def test_dump_3d_array_full_writes_file(monkeypatch, tmp_path):
    """dump_3d_array with full=True writes full-range file (nuv[0] to ir[1])."""
    from utils.constants import debug_wavelength_range_nuv, debug_wavelength_range_ir
    wmin = float(debug_wavelength_range_nuv[0])
    wmax = float(debug_wavelength_range_ir[1])
    mid = (wmin + wmax) / 2
    array = np.array([[wmin - 100, 0.0], [mid, 1.0], [wmax + 100, 0.0]])
    cfg = type("Cfg", (), {"run_nuv": True, "run_vis": True, "run_nir": True})()
    monkeypatch.setattr("utils.debug_dumps.get_global_config", lambda: cfg)

    dump_3d_array(array, tmp_path, "star", "tag", full=True)

    out = tmp_path / "star_tag_full.txt"
    assert out.exists()
    data = np.loadtxt(out)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[0] == 1
    assert np.allclose(data[0], [mid, 1.0])


def test_dump_3d_array_zoom_writes_nuv_zoom_file(monkeypatch, tmp_path):
    """dump_3d_array with zoom=True writes NUV zoom file when run_nuv is True."""
    from utils.constants import DEBUG_WL_A_NUV
    wmin, wmax = float(DEBUG_WL_A_NUV[0]), float(DEBUG_WL_A_NUV[1])
    mid = (wmin + wmax) / 2
    array = np.array([[wmin - 20, 0.0], [mid, 1.0], [wmax + 20, 0.0]])
    cfg = type("Cfg", (), {"run_nuv": True, "run_vis": False, "run_nir": False})()
    monkeypatch.setattr("utils.debug_dumps.get_global_config", lambda: cfg)

    dump_3d_array(array, tmp_path, "star", "tag", zoom=True)

    out = tmp_path / "star_tag_NUV_zoom.txt"
    assert out.exists()
    data = np.loadtxt(out)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert np.allclose(data[0], [mid, 1.0])


def test_dump_1d_for_channel_full_writes_file(tmp_path):
    """dump_1d_for_channel with full=True writes masked file for that channel (uses real NUV range)."""
    from utils.constants import debug_wavelength_range_nuv
    wmin, wmax = debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1]
    wave = np.array([wmin - 100.0, (wmin + wmax) / 2, wmax + 100.0])
    array = np.array([0.0, 1.0, 0.0])
    dump_1d_for_channel(wave, array, tmp_path, "star", "tag", "NUV", full=True)

    out = tmp_path / "star_tag_NUV.txt"
    assert out.exists()
    data = np.loadtxt(out)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape == (1, 2)
    assert np.allclose(data[0], [(wmin + wmax) / 2, 1.0])


def test_dump_1d_for_channel_zoom_writes_file(tmp_path):
    """dump_1d_for_channel with zoom=True writes masked file for that channel (uses real NUV zoom range)."""
    from utils.constants import DEBUG_WL_A_NUV
    wmin, wmax = DEBUG_WL_A_NUV[0], DEBUG_WL_A_NUV[1]
    wave = np.array([wmin - 10.0, (wmin + wmax) / 2, wmax + 10.0])
    array = np.array([0.0, 1.0, 0.0])
    dump_1d_for_channel(wave, array, tmp_path, "star", "tag", "NUV", zoom=True)

    out = tmp_path / "star_tag_NUV_zoom.txt"
    assert out.exists()
    data = np.loadtxt(out)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape == (1, 2)
    assert np.allclose(data[0], [(wmin + wmax) / 2, 1.0])
