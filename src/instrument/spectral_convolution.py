import numpy as np
import logging
from domain.star import Star
from configs.channel_config import Channel
from configs.global_config import get_global_config
from loaders.run_waltzer_context import RunContext
from instrument.wavelength_range import compute_extended_wavelength_range
from utils.debug_dumps import dump_1d_for_channel, dump_effective_area_txt, dump_npz_snapshot
from utils.flux_image_array import plot_1d_for_channel

def compute_broadened_channel_flux(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, channel: Channel, star: Star):
    # Cut up array to broaden with gauss later
    cut_photon_flux, wavelength = cut_wavelength_window_with_margin(photon_flux_at_earth, wavelengths_total, channel)
    logging.info("BG_FLUX_CUT star_id=%s channel=%s n=%d wl_min=%.1f wl_max=%.1f", star.name, channel.channel_name, wavelength.size, float(wavelength[0]), float(wavelength[-1]))

    # Gaussian Broadening of flux over wavelengths
    photon_flux_smoothed =  gaussbroad(wavelength, cut_photon_flux, channel.pixel_scale)
    logging.info("BG_FLUX_SMOOTHED star_id=%s channel=%s n=%d wl_min=%.1f wl_max=%.1f", star.name, channel.channel_name, wavelength.size, float(wavelength[0]), float(wavelength[-1]))
    return photon_flux_smoothed, wavelength


def cut_wavelength_window_with_margin(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, channel: Channel, margin_A: float = 200.0):
    wl_min_ext, wl_max_ext = compute_extended_wavelength_range([channel], margin_A)

    i0_raw = np.searchsorted(wavelengths_total, wl_min_ext)
    i1_raw = np.searchsorted(wavelengths_total, wl_max_ext, side="right")

    i0 = max(i0_raw, 0)
    i1 = min(i1_raw, len(wavelengths_total))

    wavelength_cut = wavelengths_total[i0:i1]
    flux_cut = photon_flux_at_earth[i0:i1]

    if len(wavelength_cut) == 0:
        raise ValueError(f"Channel range [{channel.effective_area_wavelength[0]}, {channel.effective_area_wavelength[-1]}] with margin {margin_A} does not overlap total wavelengths.")

    logging.info("BG_FLUX_CUT_WINDOW indices=[%d:%d] wl=%.1f-%.1f n=%d", i0, i1, float(wavelength_cut[0]), float(wavelength_cut[-1]), len(wavelength_cut))

    return flux_cut, wavelength_cut


def gaussbroad(wavelength, spectra, hwhm):
    # Smooth a spectrum by convolution with a Gaussian of specified HWHM.
    # wavelength (input vector): wavelength scale of spectrum to be smoothed
    # spectra    (input vector): spectrum to be smoothed
    # hwhm      (input scalar): half width at half maximum of smoothing Gaussian
    # Returns a vector containing the Gaussian-smoothed spectrum.
    #
    # Edit History:
    # - Dec-90 GB,GM: Rewrote with Fourier convolution algorithm.
    # - Jul-91 AL: Translated from ANA to IDL.
    # - 22-Sep-91 JAV: Relaxed constant dispersion check; vectorized, 50% faster.
    # - 05-Jul-92 JAV: Converted to function, handle nonpositive HWHM.

    # Return input spectrum if half-width is nonpositive:
    if hwhm <= 0 or len(wavelength) < 2:
            return spectra

    # Calculate (uniform) dispersion (wavelength per pixel)
    dw = (wavelength[-1] - wavelength[0]) / (len(wavelength) - 1)

    # Make smoothing gaussian# extend to 4 sigma.
    # Note: 4.0 / sqrt(2.0*numpy.log(2.0)) = 3.3972872 & sqrt(numpy.log(2.0))=0.83255461
    # sqrt(numpy.log(2.0)/pi)=0.46971864 (*1.0000632 to correct for >4 sigma wings)
    # Guard for extreme broadening: if width > 5x the total range, 
    # the result is essentially a flat line of the average flux.
    if hwhm > 5 * (wavelength[-1] - wavelength[0]): 
        # Using np.mean() is more numerically stable than sum/len and 
        # ensures we average based only on the flux data points present.
        return np.full(len(wavelength), np.mean(spectra))

    # Points in half gaussian
    nhalf = int(3.3972872*hwhm/dw)      
    # Points in gaussian (odd!)
    ng = 2 * nhalf + 1              
    # Wavelength scale of gaussian
    wg = dw * (np.arange(ng) - (ng-1)/2.0)  
    # Convenient absisca
    xg = ( (0.83255461) / hwhm) * wg        
    # Unit area gaussian w/ FWHM
    gpro = ( (0.46974832) * dw / hwhm) * np.exp(-xg*xg)
    # Force normalization to ensure flux conservation
    gpro = gpro / np.sum(gpro)

    # Pad pixels on each end to minimize impact of Fourier ringing and edge drop-off.
    # This repeats the edge values so the Gaussian doesn't "see" zeros at the boundaries.
    npad = nhalf + 2                
    # Concatenate the padded ends with the original spectra
    spad = np.concatenate((np.full(npad,spectra[0]),spectra,np.full(npad,spectra[-1])))

    # Convolve with gaussian
    sout = np.convolve(spad,gpro,mode='full')
    # Trim to original data/length
    sout = sout[npad : npad + len(wavelength)]

    # Return broadened spectrum
    return sout


def counts_per_s_px_conv_per_channel(broadened_photon_flux: np.ndarray, wavelength: np.ndarray, channel: Channel, star: Star, ctx: RunContext):
    """
    Convert photon flux at Earth [photons/cm²/s/Å] into counts/s/pixel for a single channel and gauss broaden it.
    """
    cfg = get_global_config()
    # Get the pixel wavelength grid from the channel and interpolate smoothed Earth flux onto the pixel wavelength grid
    photon_flux_on_pixel = np.interp(channel.effective_area_wavelength, wavelength, broadened_photon_flux)

    # Step 3: convert photons per Angstrom into photons per pixel
    photons_per_pixel_cm2_s = photon_flux_on_pixel * channel.pixel_scale

    # Step 4: apply effective area to get detector counts per second per pixel
    counts_s_px_convolved = photons_per_pixel_cm2_s * channel.effective_area

    if cfg.write_intermediate_arrays:
        dump_1d_for_channel(channel.effective_area_wavelength, counts_s_px_convolved, ctx.output_dir, star.name, "Detector_counts_s_px_convolved", channel.channel_name, full=True, zoom=True)
        dump_effective_area_txt(ctx.output_dir, channel.channel_name, channel.effective_area_wavelength, channel.effective_area, channel.pixel_scale)
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_convolved_counts_full.npz", counts_s_px_convolved=counts_s_px_convolved)

    if cfg.produce_flux_convolution_plots:
        plot_1d_for_channel(channel.effective_area_wavelength, counts_s_px_convolved, ctx.output_dir, star, filename_tag="Detector_counts_s_px_convolved", title_text="Convolved Counts", y_label="Counts s⁻¹ pixel⁻¹", channel_name=channel.channel_name, full=True)

    logging.info("Counts per pixel computed: channel=%s bins=%d", channel.channel_name, len(counts_s_px_convolved))
    return counts_s_px_convolved
