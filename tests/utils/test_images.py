from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from utils.images import write_image_png, write_frames_png


def _channel_for_image():
    # Minimal SpectroscopyChannel-like object for write_image_png
    return SimpleNamespace(
        channel_name="NUV",
        x_pixels=6,
        y_pixels=4,
        bias_offset=10.0,
        read_noise=3.0,
        dark_noise=0.5,
        dark_current_sigma=0.2,
        exposure_s=5.0,
    )


def _ctx_for_image(tmp_path):
    # Minimal RunContext-like object
    return SimpleNamespace(
        target_name="HD 2685",
        output_dir=tmp_path,
    )


def test_write_image_png_creates_file_with_expected_name(tmp_path):
    ctx = _ctx_for_image(tmp_path)
    ch = _channel_for_image()

    array = np.ones((ch.y_pixels, ch.x_pixels), dtype=float)

    write_image_png(array, "BIAS", ctx, ch, show_stats=True)

    expected_name = f"{ctx.target_name.replace(' ', '_')}_{ch.channel_name}_BIAS_image.png"
    expected_path = tmp_path / expected_name

    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


def test_write_image_png_without_stats(tmp_path):
    ctx = _ctx_for_image(tmp_path)
    ch = _channel_for_image()

    array = np.ones((ch.y_pixels, ch.x_pixels), dtype=float)

    write_image_png(array, "DARK", ctx, ch, show_stats=False)

    expected_name = f"{ctx.target_name.replace(' ', '_')}_{ch.channel_name}_DARK_image.png"
    expected_path = tmp_path / expected_name

    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


def _star_for_frames():
    # Minimal Star-like object for write_frames_png
    return SimpleNamespace(
        name="HD 2685",
        mass=1.1,
        distance_pc=100.0,
        effective_temperature=5778.0,
    )


def _ctx_for_frames(tmp_path):
    return SimpleNamespace(output_dir=tmp_path)


def test_write_frames_png_creates_numbered_files(tmp_path):
    ctx = _ctx_for_frames(tmp_path)
    star = _star_for_frames()

    frames = [
        np.ones((4, 6), dtype=float),
        np.ones((4, 6), dtype=float) * 2.0,
    ]

    # Minimal headers; only FILETYPE is needed for stats selection
    hdr1 = fits.Header()
    hdr1["FILETYPE"] = "BIAS"
    hdr2 = fits.Header()
    hdr2["FILETYPE"] = "BIAS"
    headers = [hdr1, hdr2]

    write_frames_png(frames, headers, frame_type="BIAS", channel_tag="NUV", ctx=ctx, star=star, show_stats=True)

    base = "WALTzER_HD_2685_NUV_BIAS_"
    paths = [
        tmp_path / f"{base}00000.png",
        tmp_path / f"{base}00001.png",
    ]

    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_write_frames_png_no_frames_does_nothing(tmp_path):
    ctx = _ctx_for_frames(tmp_path)
    star = _star_for_frames()

    write_frames_png([], [], frame_type="BIAS", channel_tag="NUV", ctx=ctx, star=star, show_stats=False)

    # Directory should remain empty
    assert not any(tmp_path.iterdir())

"""Tests for utils.images."""
import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import numpy as np
from types import SimpleNamespace

from utils.images import format_header, write_frames_png


def test_format_header_returns_key_equals_value():
    """format_header formats present key as key=value."""
    hdr = {"MEAN": 1.5}
    assert format_header(hdr, "MEAN") == "MEAN=1.50"


def test_format_header_missing_key_returns_n_a():
    """format_header returns key=n/a for missing key."""
    hdr = {}
    assert format_header(hdr, "MEAN") == "MEAN=n/a"


def test_format_header_custom_fmt_str():
    """format_header uses fmt_str for formatting."""
    hdr = {"X": 3.14159}
    assert format_header(hdr, "X", ".4f") == "X=3.1416"


def _dummy_star():
    return SimpleNamespace(name="TestStar", mass=1.0, distance_pc=10.0, effective_temperature=5778.0)


def test_write_frames_png_empty_frames_returns_without_creating_files(tmp_path):
    """write_frames_png with empty frames returns early; no PNG files created."""
    ctx = SimpleNamespace(output_dir=tmp_path)
    write_frames_png([], [], "BIAS", "NUV", ctx, _dummy_star())
    assert list(tmp_path.glob("*.png")) == []


def test_write_frames_png_writes_one_file(tmp_path):
    """write_frames_png writes one PNG file per frame."""
    ctx = SimpleNamespace(output_dir=tmp_path)
    frame = np.zeros((4, 4), dtype=float)
    hdr = {}

    write_frames_png([frame], [hdr], "bias", "NUV", ctx, _dummy_star())

    out = tmp_path / "WALTzER_TestStar_NUV_bias_00000.png"
    assert out.exists()
