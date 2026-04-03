import numpy as np
import pytest

from instrument.background_star_common import compute_roll_angle_samples, get_cached_counts, rotated_coordinates, within_rotated_bounds_mask


# Tests: compute_roll_angle_samples
# Behavior: zero offset from target yields minimum two samples along roll range (photometry channel)
def test_compute_roll_angle_samples_zero_offset_yields_two_steps(make_photometry_channel):
    ch = make_photometry_channel()
    angles = compute_roll_angle_samples(0.0, ch, 0.0, 90.0)
    assert angles.dtype == np.float32
    assert angles.shape == (2,)
    assert np.isclose(float(angles[0]), 0.0)
    assert np.isclose(float(angles[-1]), 90.0)


# Tests: compute_roll_angle_samples
# Behavior: identical start and stop still returns at least two samples at that angle (spectroscopy channel)
def test_compute_roll_angle_samples_equal_start_stop(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(pixel_scale=1.0)
    angles = compute_roll_angle_samples(10.0, ch, 5.0, 5.0)
    assert angles.shape[0] >= 2
    assert np.allclose(angles, 5.0)


# Tests: compute_roll_angle_samples
# Behavior: larger sky offset produces finer sampling for the same roll sweep (photometry channel)
def test_compute_roll_angle_samples_more_steps_for_larger_radius(make_photometry_channel):
    ch = make_photometry_channel(pixel_scale=1.0)
    small = compute_roll_angle_samples(1.0, ch, 0.0, 1.0, max_motion_per_step_px=1.0)
    large = compute_roll_angle_samples(100.0, ch, 0.0, 1.0, max_motion_per_step_px=1.0)
    assert large.shape[0] >= small.shape[0]


# Tests: compute_roll_angle_samples
# Behavior: first and last samples match roll endpoints (spectroscopy channel)
def test_compute_roll_angle_samples_endpoints_match(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(pixel_scale=0.5)
    start, stop = -2.5, 7.5
    angles = compute_roll_angle_samples(5.0, ch, start, stop)
    assert np.isclose(float(angles[0]), start, rtol=1e-5)
    assert np.isclose(float(angles[-1]), stop, rtol=1e-5)


# Tests: compute_roll_angle_samples
# Behavior: zero pixel scale divides when converting arc to pixels (photometry channel)
def test_compute_roll_angle_samples_zero_pixel_scale_raises(make_photometry_channel):
    ch = make_photometry_channel(pixel_scale=0.0)
    with pytest.raises(ZeroDivisionError):
        compute_roll_angle_samples(1.0, ch, 0.0, 10.0)


# Tests: get_cached_counts
# Behavior: returns cached value for known star/channel key
def test_get_cached_counts_returns_cached_value(make_photometry_channel, make_star_catalog):
    channel = make_photometry_channel(channel_name="NIR")
    expected = np.array([1.0, 2.0], dtype=np.float32)
    catalog = make_star_catalog(counts={("star_1", "NIR"): expected})
    result = get_cached_counts("star_1", catalog, channel, frame_index=3)
    np.testing.assert_array_equal(result, expected)


# Tests: get_cached_counts
# Behavior: returns None when star/channel key is missing
def test_get_cached_counts_returns_none_when_missing(make_photometry_channel, make_star_catalog):
    channel = make_photometry_channel(channel_name="NIR")
    catalog = make_star_catalog()
    assert get_cached_counts("missing_star", catalog, channel, frame_index=5) is None


# Tests: rotated_coordinates
# Behavior: rotates offsets with detector-frame sign convention
def test_rotated_coordinates_zero_and_ninety_deg():
    u, v = rotated_coordinates(1.0, 0.0, np.array([0.0, 90.0], dtype=np.float32))
    assert np.allclose(u, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(v, np.array([0.0, -1.0], dtype=np.float32), atol=1e-6)


# Tests: within_rotated_bounds_mask
# Behavior: marks only points inside rectangle bounds
def test_within_rotated_bounds_mask_basic():
    u = np.array([0.0, 0.4, 1.2], dtype=np.float32)
    v = np.array([0.0, 0.3, 0.2], dtype=np.float32)
    mask = within_rotated_bounds_mask(u, v, (0.5, 0.5))
    assert np.array_equal(mask, np.array([True, True, False]))
