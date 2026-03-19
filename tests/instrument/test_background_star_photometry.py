import numpy as np
from unittest.mock import patch

from instrument.background_star_photometry import (
    generate_background_star_photometry_image,
)


# Tests: generate_background_star_photometry_image
# Behavior: returns zero image when no stars are present
def test_bg_photometry_no_stars(make_photometry_channel, make_star_catalog):
    channel = make_photometry_channel(x_pixels=32, y_pixels=32)
    catalog = make_star_catalog()

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        image, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.all(image == 0.0)
    assert arcs == {}


# Tests: generate_background_star_photometry_image
# Behavior: ignores stars outside detector
def test_bg_photometry_star_outside_detector(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(x_pixels=32, y_pixels=32)
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (1000.0, 1000.0)},
        counts={("star_1", channel.channel_name): 100.0},
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        image, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.all(image == 0.0)
    assert arcs == {}


# Tests: generate_background_star_photometry_image
# Behavior: ignores stars without cached counts
def test_bg_photometry_missing_counts(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(x_pixels=32, y_pixels=32)
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        image, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.all(image == 0.0)
    assert arcs == {}


# Tests: generate_background_star_photometry_image
# Behavior: renders contribution for a single star
def test_bg_photometry_single_star(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(
        x_pixels=32,
        y_pixels=32,
        psf_image=np.ones((3, 3), dtype=float),
        psf_center_x=1,
        psf_center_y=1,
    )
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
        counts={("star_1", channel.channel_name): 100.0},
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        image, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.sum(image) > 0.0
    assert "star_1" in arcs
    assert len(arcs["star_1"]) >= 2


# Tests: generate_background_star_photometry_image
# Behavior: combines contributions from multiple stars
def test_bg_photometry_multiple_stars(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(
        x_pixels=32,
        y_pixels=32,
        psf_image=np.ones((3, 3), dtype=float),
        psf_center_x=1,
        psf_center_y=1,
    )
    catalog = make_star_catalog(
        stars={
            "star_1": make_star(gaia_magnitude=12.0),
            "star_2": make_star(gaia_magnitude=13.0),
        },
        offsets={"star_1": (0.0, 0.0), "star_2": (2.0, 0.0)},
        counts={
            ("star_1", channel.channel_name): 100.0,
            ("star_2", channel.channel_name): 50.0,
        },
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        image, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.sum(image) > 0.0
    assert set(arcs.keys()) == {"star_1", "star_2"}


# Tests: generate_background_star_photometry_image
# Behavior: returns arc positions for visible stars
def test_bg_photometry_returns_arc_positions(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(
        x_pixels=32,
        y_pixels=32,
        psf_image=np.ones((3, 3), dtype=float),
        psf_center_x=1,
        psf_center_y=1,
    )
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
        counts={("star_1", channel.channel_name): 100.0},
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        _, arcs = generate_background_star_photometry_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=10.0,
            frame_index=0,
        )

    assert "star_1" in arcs
    assert len(arcs["star_1"]) >= 2
    for x_pos, y_pos in arcs["star_1"]:
        assert 0 <= x_pos < channel.x_pixels
        assert 0 <= y_pos < channel.y_pixels


# Tests: generate_background_star_photometry_image
# Behavior: raises on invalid pixel scale to lock current invalid-input contract
def test_bg_photometry_invalid_pixel_scale_zero_raises(make_photometry_channel, make_star_catalog, make_star):
    channel = make_photometry_channel(
        x_pixels=32,
        y_pixels=32,
        pixel_scale=0.0,
        psf_image=np.ones((3, 3), dtype=float),
        psf_center_x=1,
        psf_center_y=1,
    )
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
        counts={("star_1", channel.channel_name): 100.0},
    )

    with patch(
        "instrument.background_star_photometry.get_photometry_placement",
        return_value=(16, 16.0),
    ):
        with np.testing.assert_raises(ZeroDivisionError):
            generate_background_star_photometry_image(
                channel,
                catalog,
                roll_angle_start=0.0,
                roll_angle_stop=10.0,
                frame_index=0,
            )