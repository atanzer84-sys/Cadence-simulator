import numpy as np
import logging
from domain.star import Star
from flux.flux_calc import calculateFluxOnEarth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from instrument.spectrum_spread import spread_1d_spectrum_to_2d
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, compute_broadened_channel_flux
from configs.global_config import get_global_config
from utils.constants import PHOTON_ENERGY_CONVERSION_A

def prepare_all_detector_images_all_channels(star: Star, ctx: RunContext, nuv: SpectroscopyChannel, vis: SpectroscopyChannel):
    logging.info("Starting convolution to instrument")
    print("Starting convolution to instrument")

    flux, wavelengths_total = calculateFluxOnEarth(star, ctx)

    spectra_2d_nuv = prepare_all_detector_images_one_channel(flux, wavelengths_total, nuv, ctx, star)
    spectra_2d_vis = prepare_all_detector_images_one_channel(flux, wavelengths_total, vis, ctx, star)

    return spectra_2d_nuv, spectra_2d_vis

def prepare_all_detector_images_one_channel(flux: np.ndarray, wavelengths: np.ndarray,channel: SpectroscopyChannel, ctx: RunContext, star: Star):

    cfg = get_global_config()
    # 1) flux -> photons
    # from ergs/s/cm2/A to photons/s/cm2/A
    photons_star = convert_flux_to_photons(flux, wavelengths)
    ctx.produce_plots.plot_flux_and_photons_windows(wavelengths, photons_star, ctx.output_dir, star, "FluxCalc_2_photons",  "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")
    ctx.test_mode.dump_1d_array(wavelengths, photons_star, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)

    # 2) photons -> broadened -> counts/s/pixel (single channel path, reusing existing pieces)
    broadened_flux, wavelength = compute_broadened_channel_flux(photons_star, wavelengths, channel, ctx.output_dir, cfg, star, ctx)

    counts_s_px_convolved = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, star, ctx)

    # 3) counts -> 2D
    spectra_2d = spread_1d_spectrum_to_2d(counts_s_px_convolved, channel)

    return spectra_2d


def convert_flux_to_photons(flux_unred, wavelengths):
    logging.info("Converting flux to photon flux")
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths #from ergs/s/cm2/A to photons/s/cm2/A

    logging.info(f"photon_flux_at_earth_A shape: {photon_flux.shape}")
    return photon_flux
