import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from instrument.prepare_detector_images import prepare_all_detector_images_all_channels


def _channel(name="NUV"):
    # Minimal SpectroscopyChannel-like object for prepare_detector_images tests
    return SimpleNamespace(
        channel_name=name,
        x_pixels=4,
        y_pixels=3,
    )


def _ctx(tmp_path):
    return SimpleNamespace(output_dir=tmp_path)


def _star():
    return SimpleNamespace(name="TESTSTAR")


def test_prepare_all_detector_images_calls_dependencies_and_returns_shapes(tmp_path):
    star = _star()
    ctx = _ctx(tmp_path)
    nuv = _channel("NUV")
    vis = _channel("VIS")

    photon_flux = np.array([1.0, 2.0, 3.0], dtype=float)
    wavelengths = np.array([100.0, 101.0, 102.0], dtype=float)

    fake_spectra_nuv = np.ones((nuv.y_pixels, nuv.x_pixels), dtype=float)
    fake_spectra_vis = np.ones((vis.y_pixels, vis.x_pixels), dtype=float) * 3.0

    with patch(
        "instrument.prepare_detector_images.calculateFluxOnEarth",
        return_value=(photon_flux, wavelengths),
    ) as mock_flux, patch(
        "instrument.prepare_detector_images.prepare_all_detector_images_one_channel",
        side_effect=[fake_spectra_nuv, fake_spectra_vis],
    ) as mock_one_channel:

        spectra_2d_nuv, spectra_2d_vis = prepare_all_detector_images_all_channels(
            star, ctx, nuv, vis
        )

    # calculateFluxOnEarth called once with star, ctx
    mock_flux.assert_called_once()
    args, _ = mock_flux.call_args
    assert args[0] is star
    assert args[1] is ctx

    # prepare_all_detector_images_one_channel called twice: NUV then VIS
    assert mock_one_channel.call_count == 2
    (args_nuv, _), (args_vis, _) = mock_one_channel.call_args_list
    # First call: NUV channel
    assert np.allclose(args_nuv[0], photon_flux)
    assert np.allclose(args_nuv[1], wavelengths)
    assert args_nuv[2] is nuv
    assert args_nuv[3] is ctx
    assert args_nuv[4] is star
    # Second call: VIS channel
    assert np.allclose(args_vis[0], photon_flux)
    assert np.allclose(args_vis[1], wavelengths)
    assert args_vis[2] is vis
    assert args_vis[3] is ctx
    assert args_vis[4] is star

    # Returned spectra have expected shapes and values
    assert spectra_2d_nuv.shape == (nuv.y_pixels, nuv.x_pixels)
    assert spectra_2d_vis.shape == (vis.y_pixels, vis.x_pixels)
    assert np.allclose(spectra_2d_nuv, fake_spectra_nuv)
    assert np.allclose(spectra_2d_vis, fake_spectra_vis)

