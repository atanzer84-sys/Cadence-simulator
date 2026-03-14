"""Tests for utils.images_common."""

from tests.helpers.star_factory import star
from utils import images_common


def test_format_star_metadata_includes_teff_and_distance():
    """Teff (K) and distance (pc) appear in the formatted metadata string."""
    s = star(effective_temperature=5777.6, distance_pc=42.3, gaia_magnitude=None)

    text = images_common.format_star_metadata(s)

    assert "5778 K" in text
    assert "42 pc" in text


def test_format_star_metadata_none_returns_empty_string():
    """Passing None returns an empty string."""
    assert images_common.format_star_metadata(None) == ""


def test_format_star_metadata_includes_gaia_magnitude_with_one_decimal():
    """When gaia_magnitude is set, it appears in the string formatted to one decimal."""
    s = star(gaia_magnitude=7.64)
    text = images_common.format_star_metadata(s)
    assert "7.6" in text


def test_format_frame_title_includes_metadata_when_star_given():
    """Title includes target name, channel/type, and star Teff/distance when star is provided."""
    s = star(effective_temperature=6000.0, distance_pc=50.0, gaia_magnitude=None)

    title = images_common.format_frame_title("HD 2685", "NUV", "BIAS", s)

    assert "HD 2685" in title
    assert "NUV BIAS" in title
    assert "6000 K" in title
    assert "50 pc" in title


def test_normalize_target_name_replaces_spaces_with_underscores():
    """Spaces in target names are replaced by underscores for filenames/safe IDs."""
    assert images_common.normalize_target_name("HD 2685") == "HD_2685"
    assert images_common.normalize_target_name("WASP 99 b") == "WASP_99_b"

