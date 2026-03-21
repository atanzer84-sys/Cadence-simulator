import numpy as np
import logging
from configs.global_config import get_global_config
from domain.star import Star
from flux.flux_calc import calculateFluxOnEarth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel, SpectroscopyChannel, Channel
from instrument.spectrum_spread import spread_target_star_spectrum_to_2d, get_spectrum_placement
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, compute_broadened_channel_flux
from utils.constants import PHOTON_ENERGY_CONVERSION_A
from instrument.psf_spread import spread_1d_photometry_to_2d, get_photometry_placement
from instrument.wavelength_range import get_required_wavelength_range
from utils.debug_dumps import dump_npz_snapshot, dump_cropped_image_npz, dump_1d_array
from utils.flux_image_array import plot_flux_and_photons_windows

def prepare_star_photon_flux_for_channels(star: Star, ctx: RunContext, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None):
    print("\n==== STARTING CALCULATION FOR FLUX TO INSTRUMENT =====")
    wl_min_A, wl_max_A = get_required_wavelength_range(nuv, vis, nir)
    photons_star, wavelengths_total = prepare_star_photon_flux_in_range(star, ctx, wl_min_A, wl_max_A, announce_user=True)
    return photons_star, wavelengths_total

def prepare_star_photon_flux_in_range(star: Star, ctx: RunContext, wl_min_A: float, wl_max_A: float, announce_user: bool = True):
    logging.info("Calculating flux on Earth and converting to photons for star %s", star.name)

    flux, wavelengths = calculateFluxOnEarth(star, ctx, wl_min_A, wl_max_A, announce_user=announce_user)
    photons_star = convert_flux_to_photons(flux, wavelengths)
    photons_star = np.asarray(photons_star, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)

    cfg = get_global_config()
    if cfg.write_intermediate_arrays:
        dump_1d_array(wavelengths, photons_star, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)
    if cfg.produce_flux_convolution_plots:
        plot_flux_and_photons_windows(wavelengths, photons_star, ctx.output_dir, star, "FluxCalc_photons", "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")

    return photons_star, wavelengths


def prepare_detector_image_spectroscopy(photons: np.ndarray, wavelengths: np.ndarray, channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    logging.info("Starting convolution to instrument: channel=%s mode=spectroscopy", channel.channel_name)
    print(f"\n==== STARTING CONVOLUTION TO INSTRUMENT ({channel.channel_name}) =====")

    counts_s_px_convolved = compute_counts_per_s_px_one_channel(photons, wavelengths, channel, ctx, star)
    spectra_2d = spread_target_star_spectrum_to_2d(counts_s_px_convolved, channel)

    cfg = get_global_config()
    if cfg.write_intermediate_arrays:
        placement = get_spectrum_placement(channel)
        dump_cropped_image_npz(ctx.output_dir, f"{ctx.target_name}_{channel.channel_name}_spread_image_2d.npz", spectra_2d, placement[1], half_height=500)
        if channel.spread_y_positions is not None and channel.spread_y_weights is not None and channel.spread_y_wavelengths is not None:
            dump_npz_snapshot(ctx.output_dir, f"{channel.channel_name}_spread_profile_full.npz", spread_y_positions=channel.spread_y_positions, spread_y_weights=channel.spread_y_weights, spread_y_wavelengths=channel.spread_y_wavelengths)

    logging.info("Detector image prepared: channel=%s mode=spectroscopy shape=%s", channel.channel_name, spectra_2d.shape)

    return spectra_2d

def prepare_detector_image_photometry(photons: np.ndarray, wavelengths: np.ndarray, channel: PhotometryChannel, ctx: RunContext, star: Star):
    print(f"\n==== STARTING CONVOLUTION TO INSTRUMENT ({channel.channel_name}) =====")
    logging.info("PHOTOMETRY START: channel=%s star=%s", channel.channel_name, star.name)

    counts_s_px_nir = compute_counts_per_s_px_one_channel(photons, wavelengths, channel, ctx, star)
    rate_image_e_s = spread_1d_photometry_to_2d(counts_s_px_nir, channel)

    # dump spread file and image for tests or debug reasons
    cfg = get_global_config()
    if cfg.write_intermediate_arrays:
        _, source_pixel_y = get_photometry_placement(channel)
        dump_npz_snapshot(ctx.output_dir, f"{channel.channel_name}_psf_profile_full.npz", psf_image=channel.psf_image, psf_center_x=channel.psf_center_x, psf_center_y=channel.psf_center_y, source_position_x_arcsec=channel.source_position_x_arcsec, source_position_y_arcsec=channel.source_position_y_arcsec)
        dump_cropped_image_npz(ctx.output_dir, f"{ctx.target_name}_{channel.channel_name}_spread_image.npz", rate_image_e_s, source_pixel_y, half_height=500)


    logging.info("Detector image prepared: channel=%s mode=photometry shape=%s", channel.channel_name, rate_image_e_s.shape)

    ctx.plot_star_counts_vs_noise_photometry(rate_image_e_s, channel, ctx, star)
    return rate_image_e_s

def compute_counts_per_s_px_one_channel(photons_star: np.ndarray, wavelengths: np.ndarray, channel: Channel, ctx: RunContext, star: Star):

    logging.info("Computing counts per second per pixel for channel %s", channel.channel_name)

    # 2) photons -> broadened -> counts/s/pixel (single channel path, reusing existing pieces)
    broadened_flux, wavelength = compute_broadened_channel_flux(photons_star, wavelengths, channel, star)
    counts_s_px_convolved = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, star, ctx)

    return counts_s_px_convolved

def convert_flux_to_photons(flux_unred, wavelengths):
    logging.info("Converting flux to photon flux")
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths #from ergs/s/cm2/A to photons/s/cm2/A

    logging.info(f"photon_flux_at_earth_A shape: {photon_flux.shape}")
    return photon_flux