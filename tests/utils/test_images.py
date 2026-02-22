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
    return SimpleNamespace(name="TestStar", mass=1.0, distance_pc=10.0)


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

    out = tmp_path / "WALTzER_NUV_bias_00000.png"
    assert out.exists()
