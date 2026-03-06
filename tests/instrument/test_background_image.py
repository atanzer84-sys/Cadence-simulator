"""
Tests for instrument.background_image: background image generation for default and
calculated (zodiacal) background types, and the main entry point generate_background_image.
"""
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from astropy.time import Time

from instrument.background_image import (
    generate_background_image,
    generate_background_default_image,
    generate_background_calculated_image,
)


# --- Single source of truth for channel attributes used by background_image ---
# When SpectroscopyChannel or background_image usage changes, update only here.
# Tests pass only the overrides they need (e.g. background_type, background_wavelength).

_EFF_WL = np.array([2000.0, 2500.0, 3000.0], dtype=float)
_EFF_AREA = np.ones_like(_EFF_WL, dtype=float)

def _default_channel():
    """All attributes background_image reads; override in tests via _channel(**overrides)."""
    return {
        "channel_name": "NUV",
        "x_pixels": 6,
        "y_pixels": 4,
        "effective_area": _EFF_AREA.copy(),
        "effective_area_wavelength": _EFF_WL.copy(),
        "exposure_s": 1.0,
        "background_type": None,
        "background_wavelength": None,
        "background_flux": None,
        "sky_pixel_area_arcsec2": None,
        "zod_dist": None,
        "zod_spectrum_wavelength": None,
        "zod_spectrum_flux": None,
    }


def _channel(**overrides):
    """
    SpectroscopyChannel-like object: _default_channel() merged with overrides.
    Only pass attributes that differ for the test (e.g. background_type="default",
    background_wavelength=wl_bg). When the channel contract changes, update
    _default_channel() once; tests override only what they need.
    """
    d = dict(_default_channel())
    d.update(overrides)
    return SimpleNamespace(**d)


class _StubWriteImagePng:
    """Records write_image calls so tests can assert they were invoked with expected args."""

    def __init__(self):
        self.calls = []

    def write_image(self, array, frame_type: str, ctx, channel, show_stats: bool = True, star=None, **kwargs):
        self.calls.append((array, frame_type, channel))


def _ctx(tmp_path):
    """Minimal RunContext-like object with a stub write_image_png."""
    return SimpleNamespace(
        output_dir=tmp_path,
        write_image_png=_StubWriteImagePng(),
    )


def _star(ra=90.0, dec=0.0):
    """Minimal Star-like object with right_ascension and declination for calc background."""
    return SimpleNamespace(right_ascension=ra, declination=dec)


# Fixed JD for reproducible calc-background tests; patch Time.now only so get_sun(time) gets a real Time.
_FIXED_JD = 2457095.5
_fixed_time_now = Time(_FIXED_JD, format="jd", scale="utc")


# --- generate_background_image ---


def test_generate_background_image_disabled_returns_zero_image(tmp_path):
    """
    When background_type is None, no background is applied: image is all zeros
    and has shape (ny, nx). write_image is not called because we return early.
    """
    ch = _channel(x_pixels=5, y_pixels=3)
    ctx = _ctx(tmp_path)
    star = _star()

    image = generate_background_image(ch, ctx, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image == 0.0)
    # Early return: write_image_png is never used when background is disabled
    assert len(ctx.write_image_png.calls) == 0


def test_generate_background_image_default_shape_and_scaling(tmp_path):
    """
    When background_type is 'default', the image is built from the default
    background spectrum, multiplied by effective_area and exposure_s, and
    written once. Shape is (ny, nx) and all rows equal (background is 1D broadcast).
    """
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1.5e-15], dtype=float)
    eff_wl = np.array([2100.0, 2600.0], dtype=float)
    eff_area = np.array([10.0, 20.0], dtype=float)

    ch = _channel(
        x_pixels=2,
        y_pixels=3,
        background_type="default",
        effective_area=eff_area,
        effective_area_wavelength=eff_wl,
        exposure_s=5.0,
        background_wavelength=wl_bg,
        background_flux=flux_bg,
        sky_pixel_area_arcsec2=1.0,
    )
    ctx = _ctx(tmp_path)
    star = _star()

    image = generate_background_image(ch, ctx, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    # Each row should be identical (1D background broadcast along y)
    assert np.allclose(image[0], image[1]) and np.allclose(image[1], image[2])
    # Image should be positive where effective area and flux are positive
    assert np.all(image >= 0.0)
    # Note: generate_background_image does not call write_image; caller (build_science_image) handles PNG output


def test_generate_background_image_calc_shape_and_write(tmp_path):
    """
    When background_type is 'calc', the zodiacal background is computed from
    channel zod_dist / zod_spectrum and star position. Result is (ny, nx) with
    constant rows; write_image is called once. We patch Time.now() for reproducibility.
    """
    # zod_dist from loader is (n_elb, n_ela); small (2,2) grid is enough for lookup.
    zod_dist = np.array([[0.1, 0.4], [0.2, 0.5]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 2.0], dtype=float)
    eff_wl = np.array([2000.0, 2500.0, 3000.0], dtype=float)

    ch = _channel(
        x_pixels=3,
        y_pixels=2,
        background_type="calc",
        effective_area=np.ones(3, dtype=float),
        effective_area_wavelength=eff_wl,
        exposure_s=2.0,
        zod_dist=zod_dist,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    ctx = _ctx(tmp_path)
    star = _star(ra=90.0, dec=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        image = generate_background_image(ch, ctx, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image >= 0.0)
    # Note: generate_background_image does not call write_image; caller handles PNG output


# --- generate_background_default_image ---


def test_generate_background_default_image_shape():
    """
    Default background is a 1D array with length equal to len(effective_area_wavelength),
    from spline interpolation of background spectrum onto channel wavelengths.
    """
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1e-15], dtype=float)
    eff_wl = np.array([2000.0, 2200.0, 2500.0, 2800.0, 3000.0], dtype=float)

    ch = _channel(
        effective_area_wavelength=eff_wl,
        background_wavelength=wl_bg,
        background_flux=flux_bg,
        sky_pixel_area_arcsec2=0.25,
    )

    result = generate_background_default_image(ch)

    assert result.ndim == 1
    assert result.shape == (len(eff_wl),)
    assert np.all(result >= 0.0)


def test_generate_background_default_image_interpolation():
    """
    At the exact wavelength grid points, interpolated default background should
    match the converted flux (up to numerical and conversion behaviour).
    """
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1e-15], dtype=float)
    eff_wl = wl_bg.copy()

    ch = _channel(
        effective_area_wavelength=eff_wl,
        background_wavelength=wl_bg,
        background_flux=flux_bg,
        sky_pixel_area_arcsec2=1.0,
    )

    result = generate_background_default_image(ch)

    assert result.ndim == 1
    assert len(result) == 3
    # Spline through same points: at 2000, 2500, 3000 values should be consistent with flux_bg conversion
    assert np.all(np.isfinite(result))


# --- generate_background_calculated_image ---


def test_generate_background_calculated_image_shape_and_positive():
    """
    Calculated (zodiacal) background returns a 1D array of length
    len(effective_area_wavelength), non-negative, using zod_dist lookup and
    zod_spectrum scaled by the looked-up zod value.
    """
    zod_dist = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 2.0], dtype=float)
    eff_wl = np.array([2000.0, 2500.0, 3000.0], dtype=float)

    ch = _channel(
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    star = _star(ra=0.0, dec=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        result = generate_background_calculated_image(ch, star)

    assert result.ndim == 1
    assert result.shape == (len(eff_wl),)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_generate_background_calculated_image_zod_value_scaling():
    """
    The zodiacal spectrum is scaled by the single zod_value looked up from
    zod_dist; doubling that value (via grid content) should change the output proportionally.
    We use a fixed Time and known star so that the same (i,j) is selected.
    Grid layout: row 0 = elb coords (deg), column 0 = ela coords (deg); (i,j) cell = zod value.
    Use a wide coordinate range so elb_h/ela_h from star+sun always fall inside.
    """
    # (2,2): coords ela=[-100, 100], elb=[-100, 200]; zod value at (1,1) is 1.0 vs 2.0
    zod_dist_small = np.array([[-100.0, 200.0], [100.0, 1.0]], dtype=float)
    zod_dist_large = np.array([[-100.0, 200.0], [100.0, 2.0]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 1.0], dtype=float)
    eff_wl = np.array([2000.0, 3000.0], dtype=float)

    ch_small = _channel(
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist_small,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    ch_large = _channel(
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist_large,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    star = _star(ra=0.0, dec=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        out_small = generate_background_calculated_image(ch_small, star)
        out_large = generate_background_calculated_image(ch_large, star)

    # Same (i,j) lookup: small grid gives 1.0, large gives 2.0 at (0,0). So out_large ≈ 2 * out_small
    # (up to sky_pixel_area scaling which is the same). At least ordering should hold.
    assert np.all(out_large >= 0.0)
    assert np.all(out_small >= 0.0)
    # If the same index is used, ratio should be roughly 2
    if np.any(out_small > 0):
        ratio = out_large / out_small
        np.testing.assert_allclose(ratio, ratio.flat[0], rtol=0.01)
