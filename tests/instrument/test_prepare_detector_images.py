import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import instrument.prepare_detector_images as pdi


def _channel(name="NUV"):
    # Minimal SpectroscopyChannel-like object for prepare_detector_images tests
    return SimpleNamespace(
        channel_name=name,
        x_pixels=4,
        y_pixels=3,
        effective_area_wavelength=np.array([3400.0, 5000.0, 7000.0], dtype=float),
    )


def _photometry_channel(name="NIR"):
    return SimpleNamespace(
        channel_name=name,
        x_pixels=4,
        y_pixels=3,
        effective_area_wavelength=np.array([10000.0, 14000.0, 18000.0], dtype=float),
        psf_radial_distance=np.array([0.0, 1.0]),
        psf_radial_flux=np.array([1.0, 0.5]),
    )


def _ctx(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        plot_flux_and_photons_windows=lambda *args, **kwargs: None,
        dump_1d_array=lambda *args, **kwargs: None,
    )


def _star():
    return SimpleNamespace(name="TESTSTAR")


def test_prepare_detector_image_flow_calls_dependencies_and_returns_shapes(tmp_path):
    star = _star()
    ctx = _ctx(tmp_path)
    nuv = _channel("NUV")
    vis = _channel("VIS")
    nir = _photometry_channel("NIR")

    photon_flux = np.array([1.0, 2.0, 3.0], dtype=float)
    wavelengths = np.array([100.0, 101.0, 102.0], dtype=float)

    fake_spectra_nuv = np.ones((nuv.y_pixels, nuv.x_pixels), dtype=float)
    fake_spectra_vis = np.ones((vis.y_pixels, vis.x_pixels), dtype=float) * 3.0
    fake_psf_nir = np.ones((5, 5), dtype=float) * 7.0

    with patch.object(
        pdi, "calculateFluxOnEarth", return_value=(photon_flux, wavelengths)
    ) as mock_flux, patch.object(
        pdi, "prepare_detector_image_spectroscopy",
        side_effect=[fake_spectra_nuv, fake_spectra_vis],
    ) as mock_spec, patch.object(
        pdi, "prepare_detector_image_photometry", return_value=fake_psf_nir
    ) as mock_phot:

        photons_star, wavelengths_total = pdi.prepare_star_photon_flux_for_channels(
            star, ctx, nuv, vis, nir
        )

        spectra_2d_nuv = pdi.prepare_detector_image_spectroscopy(
            photons_star, wavelengths_total, nuv, ctx, star
        )
        spectra_2d_vis = pdi.prepare_detector_image_spectroscopy(
            photons_star, wavelengths_total, vis, ctx, star
        )
        psf_nir = pdi.prepare_detector_image_photometry(
            photons_star, wavelengths_total, nir, ctx, star
        )

    # calculateFluxOnEarth called once with star, ctx
    mock_flux.assert_called_once()
    args, _ = mock_flux.call_args
    assert args[0] is star
    assert args[1] is ctx

    # Spectroscopy path called twice: NUV then VIS
    assert mock_spec.call_count == 2
    (args_nuv, _), (args_vis, _) = mock_spec.call_args_list
    # First call: NUV channel
    assert np.allclose(args_nuv[0], photons_star)
    assert np.allclose(args_nuv[1], wavelengths_total)
    assert args_nuv[2] is nuv
    assert args_nuv[3] is ctx
    assert args_nuv[4] is star
    # Second call: VIS channel
    assert np.allclose(args_vis[0], photons_star)
    assert np.allclose(args_vis[1], wavelengths_total)
    assert args_vis[2] is vis
    assert args_vis[3] is ctx
    assert args_vis[4] is star

    # Photometry path called once with NIR
    mock_phot.assert_called_once()
    args_nir, _ = mock_phot.call_args
    assert np.allclose(args_nir[0], photons_star)
    assert np.allclose(args_nir[1], wavelengths_total)
    assert args_nir[2] is nir
    assert args_nir[3] is ctx
    assert args_nir[4] is star

    # Returned spectra have expected shapes and values
    assert spectra_2d_nuv.shape == (nuv.y_pixels, nuv.x_pixels)
    assert spectra_2d_vis.shape == (vis.y_pixels, vis.x_pixels)
    assert np.allclose(spectra_2d_nuv, fake_spectra_nuv)
    assert np.allclose(spectra_2d_vis, fake_spectra_vis)
    assert np.allclose(psf_nir, fake_psf_nir)

