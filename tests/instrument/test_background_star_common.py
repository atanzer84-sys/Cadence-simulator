import numpy as np
import pytest

from instrument.background_star_common import compute_roll_angle_samples


# Tests: compute_roll_angle_samples
# Behavior: zero offset from target yields minimum two samples along roll range (photometry channel)
def test_compute_roll_angle_samples_zero_offset_yields_two_steps(make_photometry_channel):
    ch = make_photometry_channel()
    angles = compute_roll_angle_samples(0.0, 0.0, ch, 0.0, 90.0)
    assert angles.dtype == np.float32
    assert angles.shape == (2,)
    assert np.isclose(float(angles[0]), 0.0)
    assert np.isclose(float(angles[-1]), 90.0)


# Tests: compute_roll_angle_samples
# Behavior: identical start and stop still returns at least two samples at that angle (spectroscopy channel)
def test_compute_roll_angle_samples_equal_start_stop(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(pixel_scale=1.0)
    angles = compute_roll_angle_samples(10.0, 0.0, ch, 5.0, 5.0)
    assert angles.shape[0] >= 2
    assert np.allclose(angles, 5.0)


# Tests: compute_roll_angle_samples
# Behavior: larger sky offset produces finer sampling for the same roll sweep (photometry channel)
def test_compute_roll_angle_samples_more_steps_for_larger_radius(make_photometry_channel):
    ch = make_photometry_channel(pixel_scale=1.0)
    small = compute_roll_angle_samples(1.0, 0.0, ch, 0.0, 1.0, max_motion_per_step_px=1.0)
    large = compute_roll_angle_samples(100.0, 0.0, ch, 0.0, 1.0, max_motion_per_step_px=1.0)
    assert large.shape[0] >= small.shape[0]


# Tests: compute_roll_angle_samples
# Behavior: first and last samples match roll endpoints (spectroscopy channel)
def test_compute_roll_angle_samples_endpoints_match(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(pixel_scale=0.5)
    start, stop = -2.5, 7.5
    angles = compute_roll_angle_samples(3.0, 4.0, ch, start, stop)
    assert np.isclose(float(angles[0]), start, rtol=1e-5)
    assert np.isclose(float(angles[-1]), stop, rtol=1e-5)


# Tests: compute_roll_angle_samples
# Behavior: zero pixel scale divides when converting arc to pixels (photometry channel)
def test_compute_roll_angle_samples_zero_pixel_scale_raises(make_photometry_channel):
    ch = make_photometry_channel(pixel_scale=0.0)
    with pytest.raises(ZeroDivisionError):
        compute_roll_angle_samples(1.0, 0.0, ch, 0.0, 10.0)
