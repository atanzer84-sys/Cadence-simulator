import numpy as np
from unittest.mock import Mock, patch
import pytest
from instrument.prepare_detector_images import prepare_star_photon_flux_for_channels
from instrument.prepare_detector_images import prepare_star_photon_flux_in_range
from instrument.prepare_detector_images import prepare_detector_image_spectroscopy
from instrument.prepare_detector_images import prepare_detector_image_photometry
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
from instrument.prepare_detector_images import convert_flux_to_photons
from utils.constants import PHOTON_ENERGY_CONVERSION_A


# Tests: prepare_star_photon_flux_for_channels
# Behavior: uses required wavelength range for channel preparation
def test_prepare_star_photon_flux_for_channels_uses_required_range(make_star, make_run_context, make_spectroscopy_channel, make_photometry_channel):
    star = make_star()
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")
    vis = make_spectroscopy_channel(channel_name="VIS")
    nir = make_photometry_channel(channel_name="NIR")

    photons = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    wavelengths = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    with patch("instrument.prepare_detector_images.get_required_wavelength_range", return_value=(100.0, 300.0)) as mock_get_range, patch("instrument.prepare_detector_images.prepare_star_photon_flux_in_range", return_value=(photons, wavelengths)) as mock_prepare:
        photons_star, wavelengths_total = prepare_star_photon_flux_for_channels(star, ctx, nuv, vis, nir)

    assert np.array_equal(photons_star, photons)
    assert np.array_equal(wavelengths_total, wavelengths)
    mock_get_range.assert_called_once_with(nuv, vis, nir)
    mock_prepare.assert_called_once_with(star, ctx, 100.0, 300.0, announce_user=True)


# Tests: prepare_star_photon_flux_in_range
# Behavior: converts outputs to float32 and writes diagnostic outputs
def test_prepare_star_photon_flux_in_range_converts_and_calls_outputs(make_star, make_run_context):
    plot_mock = Mock()
    dump_mock = Mock()

    star = make_star(name="TestStar")
    ctx = make_run_context(
        plot_flux_and_photons_windows=plot_mock,
        dump_1d_array=dump_mock,
    )

    flux = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    wavelengths = np.array([100.0, 200.0, 300.0], dtype=np.float64)

    with patch("instrument.prepare_detector_images.calculateFluxOnEarth", return_value=(flux, wavelengths)):
        photons_star, wavelengths_total = prepare_star_photon_flux_in_range(star, ctx, 100.0, 300.0, announce_user=False)

    expected_photons = (flux * PHOTON_ENERGY_CONVERSION_A * wavelengths).astype(np.float32)

    assert photons_star.dtype == np.float32
    assert wavelengths_total.dtype == np.float32
    assert np.allclose(photons_star, expected_photons)
    assert np.allclose(wavelengths_total, wavelengths.astype(np.float32))

    plot_mock.assert_called_once()
    dump_mock.assert_called_once()


# Tests: prepare_detector_image_spectroscopy
# Behavior: returns spread image and plots detector profile
def test_prepare_detector_image_spectroscopy_returns_image_and_plots(make_star, make_run_context, make_spectroscopy_channel):
    plot_mock = Mock()

    star = make_star(name="TestStar")
    ctx = make_run_context(plot_star_counts_vs_noise_spectroscopy=plot_mock)
    channel = make_spectroscopy_channel(
        x_pixels=4,
        y_pixels=3,
        effective_area_wavelength=np.array([100.0, 200.0, 300.0, 400.0], dtype=float),
    )

    photons = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wavelengths = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    counts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    spectra_2d = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", return_value=counts) as mock_compute, patch("instrument.prepare_detector_images.spread_target_star_spectrum_to_2d", return_value=spectra_2d) as mock_spread:
        result = prepare_detector_image_spectroscopy(photons, wavelengths, channel, ctx, star)

    assert np.array_equal(result, spectra_2d)
    mock_compute.assert_called_once_with(photons, wavelengths, channel, ctx, star)
    mock_spread.assert_called_once_with(counts, channel)
    plot_mock.assert_called_once()
    plot_args = plot_mock.call_args.args

    assert np.array_equal(plot_args[0], channel.effective_area_wavelength)
    assert np.array_equal(plot_args[1], spectra_2d.max(axis=0))
    assert plot_args[2] is channel
    assert plot_args[3] is ctx
    assert plot_args[4] is star

def test_prepare_detector_image_spectroscopy_conservation(make_star, make_run_context, make_spectroscopy_channel):
    """
    Validates 2D image preparation using the exact production signature.
    Tests float32 flux conservation with a 0.01 tolerance to handle NUV drift.
    """

    # Tests: prepare_detector_image_spectroscopy
    # Behavior: conserves counts when spreading into 2D detector image

    star = make_star(name="TestStar")
    ctx = make_run_context()
    channel = make_spectroscopy_channel(
        x_pixels=64,
        y_pixels=32,
        effective_area_wavelength=np.linspace(2000, 3000, 64, dtype=np.float32),
    )

    photons = np.random.rand(channel.x_pixels).astype(np.float32) * 100.0
    wavelengths = np.linspace(2000, 3000, channel.x_pixels).astype(np.float32)

    counts = np.random.rand(channel.x_pixels).astype(np.float32) * 50.0

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", return_value=counts):
        spectra_2d = prepare_detector_image_spectroscopy(photons, wavelengths, channel, ctx, star)

    col_sums = spectra_2d.sum(axis=0)
    max_diff = np.max(np.abs(col_sums - counts))

    assert spectra_2d.shape == (channel.y_pixels, channel.x_pixels)
    assert max_diff <= 0.01

    

# Tests: prepare_detector_image_spectroscopy
# Behavior: handles zero counts without changing detector shape
def test_prepare_detector_image_spectroscopy_zero_counts(make_star, make_run_context, make_spectroscopy_channel):
    star = make_star()
    ctx = make_run_context()
    channel = make_spectroscopy_channel(
        x_pixels=4,
        y_pixels=3,
        effective_area_wavelength=np.array([100.0, 200.0, 300.0, 400.0], dtype=float),
    )

    photons = np.zeros(4, dtype=np.float32)
    wavelengths = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    counts = np.zeros(4, dtype=np.float32)
    spectra_2d = np.zeros((3, 4), dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", return_value=counts), patch("instrument.prepare_detector_images.spread_target_star_spectrum_to_2d", return_value=spectra_2d):
        result = prepare_detector_image_spectroscopy(photons, wavelengths, channel, ctx, star)

    assert result.shape == (channel.y_pixels, channel.x_pixels)
    assert np.count_nonzero(result) == 0


# Tests: prepare_detector_image_spectroscopy
# Behavior: handles single pixel detector input
def test_prepare_detector_image_spectroscopy_minimal_size(make_star, make_run_context, make_spectroscopy_channel):
    star = make_star()
    ctx = make_run_context()
    channel = make_spectroscopy_channel(
        x_pixels=1,
        y_pixels=1,
        effective_area_wavelength=np.array([100.0], dtype=float),
    )

    photons = np.array([5.0], dtype=np.float32)
    wavelengths = np.array([100.0], dtype=np.float32)
    counts = np.array([7.0], dtype=np.float32)
    spectra_2d = np.array([[7.0]], dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", return_value=counts), patch("instrument.prepare_detector_images.spread_target_star_spectrum_to_2d", return_value=spectra_2d):
        result = prepare_detector_image_spectroscopy(photons, wavelengths, channel, ctx, star)

    assert result.shape == (1, 1)
    assert result[0, 0] == 7.0


# Tests: prepare_detector_image_spectroscopy
# Behavior: propagates count preparation errors
def test_prepare_detector_image_spectroscopy_propagates_compute_error(make_star, make_run_context, make_spectroscopy_channel):
    star = make_star()
    ctx = make_run_context()
    channel = make_spectroscopy_channel()

    photons = np.array([1.0, 2.0], dtype=np.float32)
    wavelengths = np.array([100.0], dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", side_effect=ValueError):
        with pytest.raises(ValueError):
            prepare_detector_image_spectroscopy(photons, wavelengths, channel, ctx, star)


# Tests: prepare_detector_image_photometry
# Behavior: returns spread image and plots detector image
def test_prepare_detector_image_photometry_returns_image_and_plots(make_star, make_run_context, make_photometry_channel):
    plot_mock = Mock()

    star = make_star(name="TestStar")
    ctx = make_run_context(plot_star_counts_vs_noise_photometry=plot_mock)
    channel = make_photometry_channel(x_pixels=4, y_pixels=4)

    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    wavelengths = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    counts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    rate_image = np.ones((4, 4), dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", return_value=counts) as mock_compute, patch("instrument.prepare_detector_images.spread_1d_photometry_to_2d", return_value=rate_image) as mock_spread:
        result = prepare_detector_image_photometry(flux, wavelengths, channel, ctx, star)

    assert np.array_equal(result, rate_image)
    mock_compute.assert_called_once_with(flux, wavelengths, channel, ctx, star)
    mock_spread.assert_called_once_with(counts, channel, ctx)
    plot_mock.assert_called_once_with(rate_image, channel, ctx, star)


# Tests: prepare_detector_image_photometry
# Behavior: propagates count preparation errors
def test_prepare_detector_image_photometry_propagates_compute_error(make_star, make_run_context, make_photometry_channel):
    star = make_star()
    ctx = make_run_context()
    channel = make_photometry_channel()

    flux = np.array([1.0, 2.0], dtype=np.float32)
    wavelengths = np.array([100.0], dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_counts_per_s_px_one_channel", side_effect=ValueError):
        with pytest.raises(ValueError):
            prepare_detector_image_photometry(flux, wavelengths, channel, ctx, star)


# Tests: compute_counts_per_s_px_one_channel
# Behavior: passes broadened flux into count convolution
def test_compute_counts_per_s_px_one_channel_calls_dependencies(make_star, make_run_context, make_spectroscopy_channel):
    star = make_star()
    ctx = make_run_context()
    channel = make_spectroscopy_channel()

    photons_star = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    wavelengths = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    broadened_flux = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    broadened_wavelength = np.array([110.0, 210.0, 310.0], dtype=np.float32)
    counts = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    with patch("instrument.prepare_detector_images.compute_broadened_channel_flux", return_value=(broadened_flux, broadened_wavelength)) as mock_broaden, patch("instrument.prepare_detector_images.counts_per_s_px_conv_per_channel", return_value=counts) as mock_counts:
        result = compute_counts_per_s_px_one_channel(photons_star, wavelengths, channel, ctx, star)

    assert np.array_equal(result, counts)
    mock_broaden.assert_called_once_with(photons_star, wavelengths, channel, star)
    mock_counts.assert_called_once_with(broadened_flux, broadened_wavelength, channel, star, ctx)


# Tests: convert_flux_to_photons
# Behavior: converts flux to photon flux elementwise
def test_convert_flux_to_photons_multiplies_flux_and_wavelength():
    flux = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    wavelengths = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    result = convert_flux_to_photons(flux, wavelengths)

    expected = flux * PHOTON_ENERGY_CONVERSION_A * wavelengths

    assert np.allclose(result, expected)