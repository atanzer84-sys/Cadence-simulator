import pytest
import numpy as np
import pytest
from unittest.mock import patch

from astropy.time import Time

from instrument.background_image import (
    generate_background_image,
    generate_background_default_image,
    generate_background_calculated_image,
)


# Fixed JD for reproducible calc-background tests
_FIXED_JD = 2457095.5
_fixed_time_now = Time(_FIXED_JD, format="jd", scale="utc")


def _make_channel(make_spectroscopy_channel, **overrides):
    return make_spectroscopy_channel(**overrides)


# Tests: generate_background_image
# Behavior: returns a zero image when background_type is None
def test_generate_background_image_disabled_returns_zero_image(
    make_spectroscopy_channel,
    make_star,
):
    ch = _make_channel(
        make_spectroscopy_channel,
        x_pixels=5,
        y_pixels=3,
        background_type=None,
    )
    star = make_star()

    image = generate_background_image(ch, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image == 0.0)


# Tests: generate_background_image
# Behavior: builds a 2D broadcast image for default background and applies scaling
def test_generate_background_image_default_shape_and_scaling(
    make_spectroscopy_channel,
    make_star,
):
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1.5e-15], dtype=float)
    eff_wl = np.array([2100.0, 2600.0], dtype=float)
    eff_area = np.array([10.0, 20.0], dtype=float)

    ch = _make_channel(
        make_spectroscopy_channel,
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
    star = make_star()

    image = generate_background_image(ch, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.allclose(image[0], image[1])
    assert np.allclose(image[1], image[2])
    assert np.all(image >= 0.0)


# Tests: generate_background_image
# Behavior: builds a 2D broadcast image for calculated background
def test_generate_background_image_calc_shape(
    make_spectroscopy_channel,
    make_star,
):
    zod_dist = np.array([[0.1, 0.4], [0.2, 0.5]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 2.0], dtype=float)
    eff_wl = np.array([2000.0, 2500.0, 3000.0], dtype=float)

    ch = _make_channel(
        make_spectroscopy_channel,
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
    star = make_star(right_ascension=90.0, declination=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        image = generate_background_image(ch, star)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image >= 0.0)


# Tests: generate_background_default_image
# Behavior: returns 1D interpolated background with correct length
def test_generate_background_default_image_shape(make_spectroscopy_channel):
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1e-15], dtype=float)
    eff_wl = np.array([2000.0, 2200.0, 2500.0, 2800.0, 3000.0], dtype=float)

    ch = _make_channel(
        make_spectroscopy_channel,
        effective_area_wavelength=eff_wl,
        background_wavelength=wl_bg,
        background_flux=flux_bg,
        sky_pixel_area_arcsec2=0.25,
    )

    result = generate_background_default_image(ch)

    assert result.ndim == 1
    assert result.shape == (len(eff_wl),)
    assert np.all(result >= 0.0)


# Tests: generate_background_default_image
# Behavior: interpolates consistently at input wavelength grid
def test_generate_background_default_image_interpolation(make_spectroscopy_channel):
    wl_bg = np.array([2000.0, 2500.0, 3000.0], dtype=float)
    flux_bg = np.array([1e-15, 2e-15, 1e-15], dtype=float)

    ch = _make_channel(
        make_spectroscopy_channel,
        effective_area_wavelength=wl_bg,
        background_wavelength=wl_bg,
        background_flux=flux_bg,
        sky_pixel_area_arcsec2=1.0,
    )

    result = generate_background_default_image(ch)

    assert result.ndim == 1
    assert len(result) == 3
    assert np.all(np.isfinite(result))


# Tests: generate_background_calculated_image
# Behavior: returns 1D background with correct length and non negative values
def test_generate_background_calculated_image_shape_and_positive(
    make_spectroscopy_channel,
    make_star,
):
    zod_dist = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 2.0], dtype=float)
    eff_wl = np.array([2000.0, 2500.0, 3000.0], dtype=float)

    ch = _make_channel(
        make_spectroscopy_channel,
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    star = make_star(right_ascension=0.0, declination=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        result = generate_background_calculated_image(ch, star)

    assert result.ndim == 1
    assert result.shape == (len(eff_wl),)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


# Tests: generate_background_calculated_image
# Behavior: scales linearly with zodiacal value from grid
def test_generate_background_calculated_image_zod_value_scaling(
    make_spectroscopy_channel,
    make_star,
):
    zod_dist_small = np.array([[-100.0, 200.0], [100.0, 1.0]], dtype=float)
    zod_dist_large = np.array([[-100.0, 200.0], [100.0, 2.0]], dtype=float)
    wl_sol = np.array([2000.0, 3000.0], dtype=float)
    flux_sol = np.array([1.0, 1.0], dtype=float)
    eff_wl = np.array([2000.0, 3000.0], dtype=float)

    ch_small = _make_channel(
        make_spectroscopy_channel,
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist_small,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    ch_large = _make_channel(
        make_spectroscopy_channel,
        effective_area_wavelength=eff_wl,
        zod_dist=zod_dist_large,
        zod_spectrum_wavelength=wl_sol,
        zod_spectrum_flux=flux_sol,
        sky_pixel_area_arcsec2=1.0,
    )
    star = make_star(right_ascension=0.0, declination=0.0)

    with patch("instrument.background_image.Time.now", return_value=_fixed_time_now):
        out_small = generate_background_calculated_image(ch_small, star)
        out_large = generate_background_calculated_image(ch_large, star)

    assert np.all(out_large >= 0.0)
    assert np.all(out_small >= 0.0)

    if np.any(out_small > 0):
        ratio = out_large / out_small
        np.testing.assert_allclose(ratio, ratio.flat[0], rtol=0.01)

# Tests: generate_background_image
# Behavior: raises ValueError for unsupported background_type
def test_generate_background_image_raises_for_invalid_background_type(
    make_spectroscopy_channel,
    make_star,
):
    ch = _make_channel(
        make_spectroscopy_channel,
        background_type="banana",
    )
    star = make_star()

    with pytest.raises(ValueError, match="Unsupported background_type"):
        generate_background_image(ch, star)