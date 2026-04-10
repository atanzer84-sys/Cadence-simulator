import numpy as np
from unittest.mock import patch

from instrument.background_star_spectroscopy import (
    generate_background_star_spectroscopy_image,
)


# Tests: generate_background_star_spectroscopy_image
# Behavior: returns zero image and no bands when no stars contribute
def test_bg_spec_no_contribution(make_spectroscopy_channel, make_star_catalog, make_star):
    channel = make_spectroscopy_channel(x_pixels=32, y_pixels=32, channel_name="NUV")
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (1000.0, 1000.0)},
        counts={("star_1", channel.channel_name): np.array([1.0, 2.0, 3.0])},
    )

    with patch(
        "instrument.background_star_spectroscopy.get_spectrum_placement",
        return_value=(10, 16.0, 0.0, 16.0),
    ):
        image, bands = generate_background_star_spectroscopy_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.all(image == 0.0)
    assert bands == {}


# Tests: generate_background_star_spectroscopy_image
# Behavior: renders image when one star contributes
def test_bg_spec_single_star(make_spectroscopy_channel, make_star_catalog, make_star):
    channel = make_spectroscopy_channel(x_pixels=32, y_pixels=32, channel_name="NUV")
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
        counts={("star_1", channel.channel_name): np.array([1.0, 2.0, 3.0])},
    )

    def _mock_spread(image, counts_s_pixel_convolved, channel, placement, announce_user=False):
        image += np.ones_like(image, dtype=np.float32)

    with patch(
        "instrument.background_star_spectroscopy.get_spectrum_placement",
        return_value=(10, 16.0, 0.0, 16.0),
    ), patch(
        "instrument.background_star_spectroscopy.spread_1d_spectrum_to_2d",
        side_effect=_mock_spread,
    ):
        image, bands = generate_background_star_spectroscopy_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.sum(image) > 0.0
    assert bands == {}


# Tests: generate_background_star_spectroscopy_image
# Behavior: combines multiple contributing stars
def test_bg_spec_multiple_stars(make_spectroscopy_channel, make_star_catalog, make_star):
    channel = make_spectroscopy_channel(x_pixels=32, y_pixels=32, channel_name="NUV")
    catalog = make_star_catalog(
        stars={
            "star_1": make_star(gaia_magnitude=12.0),
            "star_2": make_star(gaia_magnitude=13.0),
        },
        offsets={"star_1": (0.0, 0.0), "star_2": (2.0, 0.0)},
        counts={
            ("star_1", channel.channel_name): np.array([1.0, 2.0, 3.0]),
            ("star_2", channel.channel_name): np.array([0.5, 1.0, 1.5]),
        },
    )

    def _mock_spread(image, counts_s_pixel_convolved, channel, placement, announce_user=False):
        image += np.ones_like(image, dtype=np.float32)

    with patch(
        "instrument.background_star_spectroscopy.get_spectrum_placement",
        return_value=(10, 16.0, 0.0, 16.0),
    ), patch(
        "instrument.background_star_spectroscopy.spread_1d_spectrum_to_2d",
        side_effect=_mock_spread,
    ):
        image, bands = generate_background_star_spectroscopy_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.sum(image) > 0.0
    assert bands == {}


# Tests: generate_background_star_spectroscopy_image
# Behavior: returns VIS bands when channel is VIS
def test_bg_spec_vis_bands(make_spectroscopy_channel, make_star_catalog, make_star):
    channel = make_spectroscopy_channel(
        x_pixels=32,
        y_pixels=32,
        channel_name="VIS",
        spread_half_height_pix=2,
    )
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
        counts={("star_1", channel.channel_name): np.array([1.0, 2.0, 3.0])},
    )

    def _mock_spread(image, counts_s_pixel_convolved, channel, placement, announce_user=False):
        image += np.ones_like(image, dtype=np.float32)

    with patch(
        "instrument.background_star_spectroscopy.get_spectrum_placement",
        return_value=(10, 16.0, 0.0, 16.0),
    ), patch(
        "instrument.background_star_spectroscopy.spread_1d_spectrum_to_2d",
        side_effect=_mock_spread,
    ):
        image, bands = generate_background_star_spectroscopy_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.sum(image) > 0.0
    assert "star_1" in bands
    assert "y0" in bands["star_1"]
    assert "sigma" in bands["star_1"]


# Tests: generate_background_star_spectroscopy_image
# Behavior: ignores stars without cached counts
def test_bg_spec_missing_counts(make_spectroscopy_channel, make_star_catalog, make_star):
    channel = make_spectroscopy_channel(x_pixels=32, y_pixels=32, channel_name="NUV")
    catalog = make_star_catalog(
        stars={"star_1": make_star(gaia_magnitude=12.0)},
        offsets={"star_1": (0.0, 0.0)},
    )

    with patch(
        "instrument.background_star_spectroscopy.get_spectrum_placement",
        return_value=(10, 16.0, 0.0, 16.0),
    ):
        image, bands = generate_background_star_spectroscopy_image(
            channel,
            catalog,
            roll_angle_start=0.0,
            roll_angle_stop=0.0,
            frame_index=0,
        )

    assert np.all(image == 0.0)
    assert bands == {}