"""Tests for utils.images_common."""

from types import SimpleNamespace

from utils import images_common


def test_format_star_metadata_includes_teff_and_distance():
    star = SimpleNamespace(
        effective_temperature=5777.6,
        distance_pc=42.3,
    )

    text = images_common.format_star_metadata(star)

    assert "5778 K" in text
    assert "42 pc" in text


def test_format_star_metadata_none_returns_empty_string():
    assert images_common.format_star_metadata(None) == ""


def test_format_frame_title_includes_metadata_when_star_given():
    star = SimpleNamespace(
        effective_temperature=6000.0,
        distance_pc=50.0,
    )

    title = images_common.format_frame_title("HD 2685", "NUV", "BIAS", star)

    assert "HD 2685" in title
    assert "NUV BIAS" in title
    assert "6000 K" in title
    assert "50 pc" in title


def test_normalize_target_name_replaces_spaces_with_underscores():
    assert images_common.normalize_target_name("HD 2685") == "HD_2685"
    assert images_common.normalize_target_name("WASP 99 b") == "WASP_99_b"

