from domain.star import Star
from flux.flux_calc import calculateFluxOnEarth
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from instrument.detector import counts_per_s_px_conv_all_channels
from instrument.spectrum_spread import spread_1d_spectrum_to_2d


def prepare_all_detector_images_all_channels(star: Star, ctx: RunContext, nuv: SpectroscopyChannel, vis: SpectroscopyChannel):

    photon_flux_at_earth_A, wavelengths_total = calculateFluxOnEarth(star, ctx)

    counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis = counts_per_s_px_conv_all_channels(photon_flux_at_earth_A, wavelengths_total, nuv, vis, ctx, star)

    spectra_2d_nuv = spread_1d_spectrum_to_2d(counts_s_pixel_convolved_nuv, nuv)

    spectra_2d_vis = spread_1d_spectrum_to_2d(counts_s_pixel_convolved_vis, vis)

    return spectra_2d_nuv, spectra_2d_vis