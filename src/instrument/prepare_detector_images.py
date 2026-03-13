import numpy as np
import logging
from domain.star import Star
from flux.flux_calc import calculateFluxOnEarth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel, SpectroscopyChannel, Channel
from instrument.spectrum_spread import spread_target_star_spectrum_to_2d
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, compute_broadened_channel_flux
from configs.global_config import get_global_config
from utils.constants import PHOTON_ENERGY_CONVERSION_A
from instrument.psf_spread import spread_1d_photometry_to_2d
from instrument.wavelength_range import get_required_wavelength_range

def prepare_star_photon_flux_for_channels(star: Star, ctx: RunContext, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None):
    print("\n==== STARTING CALCULATION FOR FLUX TO INSTRUMENT =====")
    wl_min_A, wl_max_A = get_required_wavelength_range(nuv, vis, nir)
    flux, wavelengths_total = calculateFluxOnEarth(star, ctx, wl_min_A, wl_max_A, announce_user=True)

    logging.info("Converting to photons")
    photons_star = convert_flux_to_photons(flux, wavelengths_total)
    ctx.plot_flux_and_photons_windows(wavelengths_total, photons_star, ctx.output_dir, star, "FluxCalc_photons", "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")
    ctx.dump_1d_array(wavelengths_total, photons_star, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)

    photons_star = np.asarray(photons_star, dtype=np.float32)
    wavelengths_total = np.asarray(wavelengths_total, dtype=np.float32)
    return photons_star, wavelengths_total

def prepare_detector_image_spectroscopy(photons: np.ndarray, wavelengths: np.ndarray, channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    logging.info("Starting convolution to instrument: channel=%s mode=spectroscopy", channel.channel_name)
    print(f"\n==== STARTING CONVOLUTION TO INSTRUMENT ({channel.channel_name})=====")

    counts_s_px_convolved = compute_counts_per_s_px_one_channel(photons, wavelengths, channel, ctx, star)
    spectra_2d = spread_target_star_spectrum_to_2d(counts_s_px_convolved, channel)

    logging.info("Detector image prepared: channel=%s mode=spectroscopy shape=%s", channel.channel_name, spectra_2d.shape)
    return spectra_2d

def prepare_detector_image_photometry(flux: np.ndarray, wavelengths: np.ndarray, channel: PhotometryChannel, ctx: RunContext, star: Star):
    print(f"\n==== STARTING CONVOLUTION TO INSTRUMENT ({channel.channel_name})=====")
    logging.info("PHOTOMETRY START: channel=%s star=%s", channel.channel_name, star.name)

    counts_s_px_nir = compute_counts_per_s_px_one_channel(flux, wavelengths, channel, ctx, star)
    rate_image_e_s = spread_1d_photometry_to_2d(counts_s_px_nir, channel, ctx)
    logging.info("Detector image prepared: channel=%s mode=photometry shape=%s", channel.channel_name, rate_image_e_s.shape)
    return rate_image_e_s

def compute_counts_per_s_px_one_channel(photons_star: np.ndarray, wavelengths: np.ndarray, channel: Channel, ctx: RunContext, star: Star):

    logging.info("Computing counts per second per pixel for channel %s", channel.channel_name)
    cfg = get_global_config()

    # 2) photons -> broadened -> counts/s/pixel (single channel path, reusing existing pieces)
    broadened_flux, wavelength = compute_broadened_channel_flux(photons_star, wavelengths, channel, ctx.output_dir, cfg, star, ctx)
    counts_s_px_convolved = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, star, ctx)

    return counts_s_px_convolved

def convert_flux_to_photons(flux_unred, wavelengths):
    logging.info("Converting flux to photon flux")
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths #from ergs/s/cm2/A to photons/s/cm2/A

    logging.info(f"photon_flux_at_earth_A shape: {photon_flux.shape}")
    return photon_flux