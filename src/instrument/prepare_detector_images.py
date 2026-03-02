import numpy as np
import logging
from domain.star import Star
from flux.flux_calc import calculateFluxOnEarth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel, SpectroscopyChannel, Channel
from instrument.spectrum_spread import spread_1d_spectrum_to_2d
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, compute_broadened_channel_flux
from configs.global_config import get_global_config
from utils.constants import PHOTON_ENERGY_CONVERSION_A
from instrument.psf_spread import build_psf_stamp_from_radial_profile, paste_stamp_center

def prepare_all_detector_images_all_channels(star: Star, ctx: RunContext, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel):
    print("==== STARTING CALCULATION FOR FLUX TO INSTRUMENT =====")
    flux, wavelengths_total = calculateFluxOnEarth(star, ctx)

    logging.info("Starting convolution to instrument")
    print("==== STARTING CONVOLUTION TO INSTRUMENT =====")
    spectra_2d_nuv = prepare_all_detector_images_spectroscopy(flux, wavelengths_total, nuv, ctx, star)
    spectra_2d_vis = prepare_all_detector_images_spectroscopy(flux, wavelengths_total, vis, ctx, star)

    # NIR Channel = Photometry
    nir_rate_frame = prepare_detector_image_photometry(flux, wavelengths_total, nir, ctx, star)
    
    return spectra_2d_nuv, spectra_2d_vis, nir_rate_frame

def prepare_all_detector_images_spectroscopy(flux: np.ndarray, wavelengths: np.ndarray, channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    counts_s_px_convolved = compute_counts_per_s_px_one_channel(flux, wavelengths, channel, ctx, star)
    spectra_2d = spread_1d_spectrum_to_2d(counts_s_px_convolved, channel)
    return spectra_2d

def prepare_detector_image_photometry(flux: np.ndarray, wavelengths: np. ndarray, channel: PhotometryChannel, ctx: RunContext, star: Star):
    logging.info("=== PHOTOMETRY START: channel=%s star=%s ===",
                 channel.channel_name, star.name)

    counts_s_px_nir = compute_counts_per_s_px_one_channel(flux, wavelengths, channel, ctx, star)
    source_e, npix = compute_photometry_signal_parameters(counts_s_px_nir, channel)

    psf = build_psf_stamp_from_radial_profile(channel.psf_radial_distance, channel.psf_radial_flux, npix)

    psf*=source_e

    nir_rate_frame = np.zeros((channel.y_pixels, channel.x_pixels), dtype=float)
    cx = channel.x_pixels // 2
    cy = channel.y_pixels // 2

    paste_stamp_center(nir_rate_frame, psf, cx, cy)

    ctx.write_image_png.write_image(nir_rate_frame, "nir_rate_frame_only", ctx, channel)

    return nir_rate_frame



def compute_counts_per_s_px_one_channel(flux: np.ndarray, wavelengths: np.ndarray, channel: Channel, ctx: RunContext, star: Star):

    logging.info("Computing counts per second per pixel for channel %s", channel.channel_name)
    cfg = get_global_config()

    photons_star = convert_flux_to_photons(flux, wavelengths)
    ctx.produce_plots.plot_flux_and_photons_windows(wavelengths, photons_star, ctx.output_dir, star, "FluxCalc_2_photons", "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")
    ctx.test_mode.dump_1d_array(wavelengths, photons_star, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)
    
    # 2) photons -> broadened -> counts/s/pixel (single channel path, reusing existing pieces)
    broadened_flux, wavelength = compute_broadened_channel_flux(photons_star, wavelengths, channel, ctx.output_dir, cfg, star, ctx)
    counts_s_px_convolved = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, star, ctx)

    return counts_s_px_convolved

def convert_flux_to_photons(flux_unred, wavelengths):
    logging.info("Converting flux to photon flux")
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths #from ergs/s/cm2/A to photons/s/cm2/A

    logging.info(f"photon_flux_at_earth_A shape: {photon_flux.shape}")
    return photon_flux

def compute_photometry_signal_parameters(counts_s_px: np.ndarray, channel:  PhotometryChannel,) -> tuple[float, float, float, float, int]:
    """
    From convolved counts per wavelength bin, compute:
        - band-integrated rate (e-/s)
        - total electrons for exposure
        - aperture radius (pix)
        - Gaussian sigma (pix)
        - PSF half-size npix
    """

    # Collapse to scalar band rate
    source_flux_s = float(np.sum(counts_s_px))

    # Aperture photometry
    radius = 0.5 * channel.aperture_pix
    # For a Gaussian profile, assume that the standard deviation is a
    # third of the aperture radius
    sigma = radius / 3.0

    # Map up to 5*sigma away from the center
    npix = int(5.0 * sigma)
    logging.info("Photometry: aperture_pix=%.3f radius=%.3f sigma=%.3f npix=%d stamp_size=%dx%d", channel.aperture_pix, radius, sigma, npix, 2*npix+1, 2*npix+1)
    return source_flux_s, npix