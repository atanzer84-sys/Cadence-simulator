"""
Tests for instrument.psf_spread: photometry PSF placement, stamp pasting,
spread_1d_photometry_to_2d, and aperture photometry.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from tests.helpers.channel_factory import photometry_channel
from instrument.psf_spread import (
    get_photometry_placement,
    paste_psf_stamp,
    spread_1d_photometry_to_2d,
    compute_aperture_photometry,
)


# -----------------------------------------------------------------------------
# get_photometry_placement
# -----------------------------------------------------------------------------


def test_get_photometry_placement_center_on_axis():
    """With source at (0,0) arcsec, placement is the detector center (x_pixels//2, y_pixels//2)."""
    ch = SimpleNamespace(
        x_pixels=10,
        y_pixels=8,
        pixel_scale=1.0,
        source_position_x_arcsec=0.0,
        source_position_y_arcsec=0.0,
    )
    x, y = get_photometry_placement(ch)
    assert x == 5
    assert y == 4


def test_get_photometry_placement_with_offset_arcsec():
    """Source offset in arcsec is converted to pixels using pixel_scale and added to center."""
    ch = SimpleNamespace(
        x_pixels=100,
        y_pixels=100,
        pixel_scale=0.5,
        source_position_x_arcsec=2.0,
        source_position_y_arcsec=-1.5,
    )
    x, y = get_photometry_placement(ch)
    # center (50, 50) + (2/0.5, -1.5/0.5) = (50+4, 50-3) = (54, 47)
    assert x == 54
    assert y == 47


def test_get_photometry_placement_none_offset_treated_as_zero():
    """When source_position_*_arcsec is None, it is treated as 0.0."""
    ch = SimpleNamespace(
        x_pixels=6,
        y_pixels=4,
        pixel_scale=1.0,
        source_position_x_arcsec=None,
        source_position_y_arcsec=None,
    )
    x, y = get_photometry_placement(ch)
    assert x == 3
    assert y == 2


# -----------------------------------------------------------------------------
# paste_psf_stamp
# -----------------------------------------------------------------------------


def test_paste_psf_stamp_fully_inside_frame():
    """Stamp fully inside frame: exact region is updated by adding stamp values."""
    frame = np.zeros((10, 10), dtype=np.float32)
    stamp = np.ones((3, 3), dtype=np.float32) * 2.0
    # Place stamp so its center is at (4, 4); psf_center at (1,1) → stamp left-top at (3,3)
    paste_psf_stamp(frame, stamp, detector_center_x=4, detector_center_y=4, psf_center_x=1, psf_center_y=1)
    assert frame[3, 3] == 2.0
    assert frame[5, 5] == 2.0
    assert np.sum(frame) == 9 * 2.0


def test_paste_psf_stamp_partial_overlap_clips_correctly():
    """Stamp partly outside frame: only overlapping pixels are added; no out-of-bounds write."""
    frame = np.zeros((4, 4), dtype=np.float32)
    stamp = np.ones((3, 3), dtype=np.float32)
    # Center at (0,0) with psf_center (1,1) → stamp left-top at (-1,-1), so stamp spans (-1,-1) to (2,2); only (0,0)-(1,1) in frame
    paste_psf_stamp(frame, stamp, detector_center_x=0, detector_center_y=0, psf_center_x=1, psf_center_y=1)
    assert frame[0, 0] == 1.0
    assert frame[1, 1] == 1.0
    assert frame[3, 3] == 0.0
    assert np.sum(frame) == 4.0  # 2x2 overlap


def test_paste_psf_stamp_no_overlap_does_nothing():
    """Stamp completely outside frame: no pixels written, frame unchanged."""
    frame = np.zeros((5, 5), dtype=np.float32)
    stamp = np.ones((2, 2), dtype=np.float32)
    # Place stamp so it lies entirely to the left of frame (stamp_right <= 0)
    paste_psf_stamp(frame, stamp, detector_center_x=0, detector_center_y=0, psf_center_x=2, psf_center_y=2)
    assert np.all(frame == 0.0)


def test_paste_psf_stamp_adds_to_existing_values():
    """Stamp is added (+=) to frame, so existing frame values are preserved and incremented."""
    frame = np.ones((5, 5), dtype=np.float32)
    stamp = np.ones((2, 2), dtype=np.float32)
    paste_psf_stamp(frame, stamp, detector_center_x=2, detector_center_y=2, psf_center_x=0, psf_center_y=0)
    # Overlap region (2:4, 2:4) was 1.0 each, add 1.0 → 2.0
    assert frame[2, 2] == 2.0
    assert frame[0, 0] == 1.0


# -----------------------------------------------------------------------------
# spread_1d_photometry_to_2d
# -----------------------------------------------------------------------------


def _channel_for_spread(psf_image=None, psf_center_x=None, psf_center_y=None, source_x=0.0, source_y=0.0):
    """Minimal channel-like object for spread_1d_photometry_to_2d (SimpleNamespace)."""
    return SimpleNamespace(
        channel_name="NIR",
        x_pixels=20,
        y_pixels=20,
        pixel_scale=1.0,
        source_position_x_arcsec=source_x,
        source_position_y_arcsec=source_y,
        psf_image=psf_image,
        psf_center_x=psf_center_x,
        psf_center_y=psf_center_y,
    )


def test_spread_1d_photometry_to_2d_success(tmp_path):
    """With valid channel and 1D counts, returns 2D detector image with correct shape and non-zero sum."""
    psf = np.ones((5, 5), dtype=np.float32) / 25.0  # normalized-like
    ch = _channel_for_spread(psf_image=psf, psf_center_x=2, psf_center_y=2)
    ctx = SimpleNamespace(output_dir=tmp_path)
    counts_1d = np.array([10.0, 20.0, 30.0], dtype=np.float32)  # total 60 e/s
    with patch("instrument.psf_spread.announce"):
        out = spread_1d_photometry_to_2d(counts_1d, ch, ctx, announce_user=False)
    assert out.shape == (20, 20)
    assert np.sum(out) == pytest.approx(60.0)


def test_spread_1d_photometry_to_2d_off_axis_raises_not_implemented_error(tmp_path):
    """Off-axis source (non-zero source_position_*_arcsec) raises NotImplementedError."""
    ch = _channel_for_spread(psf_image=np.ones((3, 3)), psf_center_x=1, psf_center_y=1, source_x=1.0, source_y=0.0)
    ctx = SimpleNamespace(output_dir=tmp_path)
    with patch("instrument.psf_spread.announce"):
        with pytest.raises(NotImplementedError, match="off-axis photometry"):
            spread_1d_photometry_to_2d(np.array([1.0]), ch, ctx, announce_user=False)


def test_spread_1d_photometry_to_2d_psf_image_none_raises_value_error(tmp_path):
    """Channel with psf_image=None raises ValueError('PSF image not loaded')."""
    ch = _channel_for_spread(psf_image=None, psf_center_x=1, psf_center_y=1)
    ctx = SimpleNamespace(output_dir=tmp_path)
    with patch("instrument.psf_spread.announce"):
        with pytest.raises(ValueError, match="PSF image not loaded"):
            spread_1d_photometry_to_2d(np.array([1.0]), ch, ctx, announce_user=False)


def test_spread_1d_photometry_to_2d_psf_center_none_raises_value_error(tmp_path):
    """Channel with psf_center_x or psf_center_y None raises ValueError('PSF center not loaded')."""
    ch = _channel_for_spread(psf_image=np.ones((3, 3)), psf_center_x=None, psf_center_y=1)
    ctx = SimpleNamespace(output_dir=tmp_path)
    with patch("instrument.psf_spread.announce"):
        with pytest.raises(ValueError, match="PSF center not loaded"):
            spread_1d_photometry_to_2d(np.array([1.0]), ch, ctx, announce_user=False)


# -----------------------------------------------------------------------------
# compute_aperture_photometry
# -----------------------------------------------------------------------------


def test_compute_aperture_photometry_non_photometry_channel_returns_none():
    """When channel is not a PhotometryChannel instance, returns None."""
    ch = SimpleNamespace(
        x_pixels=10,
        y_pixels=10,
        pixel_scale=1.0,
        source_position_x_arcsec=0.0,
        source_position_y_arcsec=0.0,
        psf_image=np.ones((3, 3)),
    )
    image = np.zeros((10, 10))
    result = compute_aperture_photometry(image, ch)
    assert result is None


def test_compute_aperture_photometry_empty_annulus_returns_none():
    """When annulus has zero pixels (e.g. image too small or source off-frame), returns None."""
    tiny = np.zeros((2, 2))
    ch_tiny = photometry_channel(x_pixels=2, y_pixels=2, psf_shape=(2, 2))
    result = compute_aperture_photometry(tiny, ch_tiny)
    assert result is None


def test_compute_aperture_photometry_returns_five_tuple():
    """With valid PhotometryChannel and image large enough for circle and annulus, returns (counts_star, x0, y0, r_inner, r_outer)."""
    ch = photometry_channel(x_pixels=32, y_pixels=32, psf_shape=(4, 4))
    image = np.zeros((32, 32), dtype=np.float64)
    # Put 100 counts in a small central region (inside aperture circle)
    image[14:18, 14:18] = 10.0  # 16 pixels * 10 = 160
    # Put uniform background in annulus so C_star ≈ circle - scaled annulus mean * Nc
    # Annulus inner R = 2*2=4, outer R = 8. Center at (16,16). Fill annulus with 1.0
    yy, xx = np.ogrid[:32, :32]
    r2 = (xx - 16) ** 2 + (yy - 16) ** 2
    annulus_mask = (r2 > 4**2) & (r2 <= 8**2)
    image[annulus_mask] = 1.0
    result = compute_aperture_photometry(image, ch)
    assert result is not None
    counts_star, x0, y0, r_inner, r_outer = result
    assert x0 == 16
    assert y0 == 16
    assert r_inner == 4.0
    assert r_outer == 8.0
    assert isinstance(counts_star, (int, float))
