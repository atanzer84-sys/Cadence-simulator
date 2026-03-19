import numpy as np
from unittest.mock import patch

from instrument.cosmic_image import generate_cosmic_rays


# Tests: test_generate_cosmic_rays_zero_rays
# Behavior: No rays produce an all-zero image
def test_generate_cosmic_rays_zero_rays(make_spectroscopy_channel, make_global_config):
    ch = make_spectroscopy_channel(x_pixels=8, y_pixels=6)
    cfg = make_global_config(
        cosmic_rays_min=0,
        cosmic_rays_max=0,
    )

    with patch('instrument.cosmic_image._rng', np.random.default_rng(42)):
        image = generate_cosmic_rays(ch, cfg)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image == 0.0)


# Tests: test_generate_cosmic_rays_signal_matches_config
# Behavior: Generated signal matches configured charge and bounds
def test_generate_cosmic_rays_signal_matches_config(make_spectroscopy_channel, make_global_config):
    ch = make_spectroscopy_channel(x_pixels=8, y_pixels=6)
    cfg = make_global_config(
        cosmic_rays_min=3,
        cosmic_rays_max=3,
        cosmic_ray_signal_electrons=50,
        cosmic_ray_length_min_px=2,
        cosmic_ray_length_max_px=4,
    )

    with patch('instrument.cosmic_image._rng', np.random.default_rng(42)):
        image = generate_cosmic_rays(ch, cfg)

    assert image.shape == (ch.y_pixels, ch.x_pixels)

    non_zero = image[image > 0]
    assert non_zero.size > 0
    assert set(np.unique(non_zero)) == {cfg.cosmic_ray_signal_electrons}

    total_signal = float(non_zero.sum())
    expected_max = 3 * cfg.cosmic_ray_signal_electrons * cfg.cosmic_ray_length_max_px

    assert total_signal <= expected_max
    assert total_signal % cfg.cosmic_ray_signal_electrons == 0


# Tests: test_generate_cosmic_rays_small_detector_edge_case
# Behavior: Small detector stays within bounds and does not crash
def test_generate_cosmic_rays_small_detector_edge_case(make_spectroscopy_channel, make_global_config):
    ch = make_spectroscopy_channel(x_pixels=1, y_pixels=1)
    cfg = make_global_config(
        cosmic_rays_min=1,
        cosmic_rays_max=1,
        cosmic_ray_signal_electrons=123,
        cosmic_ray_length_min_px=5,
        cosmic_ray_length_max_px=5,
    )

    with patch('instrument.cosmic_image._rng', np.random.default_rng(42)):
        image = generate_cosmic_rays(ch, cfg)

    assert image.shape == (1, 1)
    assert image[0, 0] in (0.0, cfg.cosmic_ray_signal_electrons)