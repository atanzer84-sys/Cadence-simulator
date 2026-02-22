"""Tests for utils.debug_dumps."""
import numpy as np
import pytest

from utils.debug_dumps import dump_1d_for_channel, dump_masked_1d


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


def test_dump_1d_for_channel_invalid_channel_raises():
    """dump_1d_for_channel raises for channel_name not NUV/VIS/IR."""
    wave = np.array([1.0, 2.0])
    array = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="cal_name"):
        dump_1d_for_channel(wave, array, "/tmp", "star", "tag", "INVALID")
