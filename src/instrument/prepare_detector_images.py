import numpy as np
import logging
from configs.global_config import get_global_config
from domain.star import Star
from flux.flux_calc import calculate_flux_on_earth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel, SpectroscopyChannel, Channel
from instrument.spectrum_spread import spread_target_star_spectrum_to_2d
from instrument.spectral_convolution import counts_per_s_px_conv_per_channel, compute_broadened_channel_flux
from instrument.wavelength_range import get_required_wavelength_range
from instrument.psf_spread import spread_1d_photometry_to_2d
from utils.debug_dumps import dump_1d_for_channel, dump_effective_area_txt, dump_npz_snapshot
from utils.flux_image_array import plot_1d_for_channel
from utils.helpers import announce

def calculate_photon_flux_density_on_Earth(star: Star, ctx: RunContext, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None, announce_user: bool = True, background_star: bool = False):
    announce(f"\n==== STARTING CALCULATION FOR FLUX TO INSTRUMENT =====", announce_user)
    
    wl_min_A, wl_max_A = get_required_wavelength_range(nuv, vis, nir)
    photons_star, wavelengths = calculate_flux_on_earth(star, ctx, wl_min_A, wl_max_A, announce_user=announce_user, background_star=background_star)

    return photons_star, wavelengths


def prepare_detector_image_spectroscopy(photons: np.ndarray, wavelengths: np.ndarray, channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    logging.info("Starting convolution to instrument: channel=%s mode=spectroscopy", channel.channel_name)
    print(f"\n==== STARTING CONVOLUTION TO INSTRUMENT ({channel.channel_name}) =====")

    counts_s_px_convolved = compute_counts_per_s_px_one_channel(photons, wavelengths, channel, ctx, star)
    spectra_2d = spread_target_star_spectrum_to_2d(counts_s_px_convolved, channel)

    cfg = get_global_config()
    if cfg.write_intermediate_arrays:
        dump_npz_snapshot(ctx.output_dir, f"{ctx.target_name}_{channel.channel_name}_spread_image_2d_full.npz", image_full=spectra_2d)
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
        dump_npz_snapshot(ctx.output_dir, f"{channel.channel_name}_psf_profile_full.npz", psf_image=channel.psf_image, psf_center_x=channel.psf_center_x, psf_center_y=channel.psf_center_y, source_position_x_arcsec=channel.source_position_x_arcsec, source_position_y_arcsec=channel.source_position_y_arcsec)
        dump_npz_snapshot(ctx.output_dir, f"{ctx.target_name}_{channel.channel_name}_spread_image_full.npz", image_full=rate_image_e_s) 


    logging.info("Detector image prepared: channel=%s mode=photometry shape=%s", channel.channel_name, rate_image_e_s.shape)

    return rate_image_e_s

def compute_counts_per_s_px_one_channel(photons_star: np.ndarray, wavelengths: np.ndarray, channel: Channel, ctx: RunContext, star: Star, background_star: bool = False):

    # 2) photons -> broadened -> counts/s/pixel (single channel path, reusing existing pieces)
    broadened_flux, wavelength = compute_broadened_channel_flux(photons_star, wavelengths, channel)
    counts_s_px_convolved = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel)
    _dump_convolved_counts(ctx, star, channel, counts_s_px_convolved, background_star=background_star)
    return counts_s_px_convolved

def _dump_convolved_counts(ctx: RunContext, star: Star, channel: Channel, counts_s_px_convolved: np.ndarray, background_star: bool = False) -> None:
    cfg = get_global_config()
    dump_arrays = cfg.write_intermediate_arrays and not background_star
    dump_plots = cfg.produce_flux_convolution_plots and not background_star
    if dump_arrays:
        dump_1d_for_channel(channel.effective_area_wavelength, counts_s_px_convolved, ctx.output_dir, star.name, "Detector_counts_s_px_convolved", channel.channel_name, full=True, zoom=True)
        dump_effective_area_txt(ctx.output_dir, channel.channel_name, channel.effective_area_wavelength, channel.effective_area, channel.pixel_scale)
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_convolved_counts_full.npz", counts_s_px_convolved=counts_s_px_convolved)
    if dump_plots:
        plot_1d_for_channel(channel.effective_area_wavelength, counts_s_px_convolved, ctx.output_dir, star, filename_tag="Detector_counts_s_px_convolved", title_text="Convolved Counts", y_label=r"Counts s$^{-1}$ pixel$^{-1}$", channel_name=channel.channel_name, full=True)

        noise_floor = channel.bias_offset + channel.dark_current * channel.exposure_s
        total_counts = counts_s_px_convolved * channel.exposure_s + noise_floor

        plot_1d_for_channel(channel.effective_area_wavelength, total_counts, ctx.output_dir, star, filename_tag="Detector_counts_s_px_convolved_noise_floor", title_text=f"Simulated Pixel Values ({channel.exposure_s:.0f} s) — bias + dark + signal", y_label=r"Counts pixel$^{-1}$", channel_name=channel.channel_name, full=True, noise_floor=noise_floor)

