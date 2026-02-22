import numpy as np
import logging
from configs.global_config import get_global_config
from utils.debug_dumps import dump_1d_for_channel
from utils.images import plot_1d_for_channel
from domain.star import Star
from configs.channel import SpectroscopyChannel
from loaders.run_waltzer_context import RunContext

# Tag strings for dump/plot output; tests import these for assertions
DUMP_TAG_CUT_WINDOW = "Detector_1_cut_wavelength_window"
DUMP_TAG_COUNTS = "Detector_2_counts_s_px_convolved"
PLOT_TAG_GAUSSBROAD = "Detector_2_gaussbroad"
PLOT_TAG_COUNTS = "counts_s_px_convolved"


def counts_per_s_px_conv_all_channels(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, ctx: RunContext, star: Star):
    logging.info("Starting convolution to instrument")
    print("Starting convolution to instrument")
    cfg = get_global_config()

    broadened_flux_nuv, wavelength_nuv = compute_broadened_channel_flux(photon_flux_at_earth, wavelengths_total, nuv, ctx.output_dir, cfg, star)
    broadened_flux_vis, wavelength_vis = compute_broadened_channel_flux(photon_flux_at_earth, wavelengths_total, vis, ctx.output_dir, cfg, star)

    counts_s_px_convolved_nuv = counts_per_s_px_conv_per_channel(broadened_flux_nuv, wavelength_nuv, nuv, ctx.output_dir, cfg, star)
    counts_s_px_convolved_vis = counts_per_s_px_conv_per_channel(broadened_flux_vis, wavelength_vis, vis, ctx.output_dir, cfg, star)

    return counts_s_px_convolved_nuv, counts_s_px_convolved_vis


def compute_broadened_channel_flux(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, channel: SpectroscopyChannel, output_dir, cfg, star: Star):

    # Cut up array to broaden with gauss later
    cut_photon_flux, wavelength = cut_wavelength_window_with_margin(photon_flux_at_earth, wavelengths_total, channel, output_dir, cfg, star)

    # Gaussian Broadening of flux over wavelengths
    photon_flux_smoothed =  gaussbroad(wavelength, cut_photon_flux, channel.pixel_scale)

    if cfg.produce_Plots:
        plot_1d_for_channel(wavelength, photon_flux_smoothed, output_dir, star, filename_tag=PLOT_TAG_GAUSSBROAD, title_text="Photon Flux after Gaussian Broadening", y_label="Photon flux [photons s⁻¹ cm⁻² Å⁻¹]", channel_name=channel.channel_name, full=True)

    logging.info("Channel %s photon_flux_smoothed sum=%g mean=%g min=%g max=%g", channel.channel_name, photon_flux_smoothed.sum(), photon_flux_smoothed.mean(), photon_flux_smoothed.min(), photon_flux_smoothed.max())

    return photon_flux_smoothed, wavelength


def cut_wavelength_window_with_margin(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, channel: SpectroscopyChannel, output_dir, cfg, star: Star, margin_A: float = 200.0):
    wl_min = channel.wavelength[0]
    wl_max = channel.wavelength[-1]

    logging.info("Cutting wavelength window: Detector wl_min=%g wl_max=%g", wl_min, wl_max)

    wl_min_ext = wl_min - margin_A
    wl_max_ext = wl_max + margin_A

    logging.info("Cutting wavelength window: Detector wl_min_ext=%g wl_max_ext=%g", wl_min_ext, wl_max_ext)

    i0_raw = np.searchsorted(wavelengths_total, wl_min_ext)
    i1_raw = np.searchsorted(wavelengths_total, wl_max_ext)

    i0 = max(i0_raw, 0)
    i1 = min(i1_raw, len(wavelengths_total))

    wavelength_cut = wavelengths_total[i0:i1]
    flux_cut = photon_flux_at_earth[i0:i1]

    if len(wavelength_cut) == 0:
        raise ValueError(
            f"Channel wavelength range [{wl_min}, {wl_max}] (extended by margin {margin_A} Å) "
            f"does not overlap wavelengths_total [{wavelengths_total[0]}, {wavelengths_total[-1]}]"
        )

    logging.info("indices after clamping: i0=%d i1=%d", i0, i1)
    logging.info("wavelengths_total[i0]=%g wavelengths_total[i1-1]=%g", wavelengths_total[i0], wavelengths_total[i1 - 1])
    logging.info("cut size=%d first_wl=%g last_wl=%g cut size flux=%d first_flux=%g last_flux=%g", len(wavelength_cut), wavelength_cut[0], wavelength_cut[-1], len(flux_cut), flux_cut[0], flux_cut[-1])

    if cfg.test_mode:
        dump_1d_for_channel(wavelength_cut, flux_cut, output_dir, star.name, DUMP_TAG_CUT_WINDOW, channel_name=channel.channel_name, full=True, zoom=True)
    if cfg.produce_Plots:
        plot_1d_for_channel(wavelength_cut, flux_cut, output_dir, star, filename_tag=DUMP_TAG_CUT_WINDOW, title_text="Photon Flux before Gaussian Broadening", y_label="Photon flux [photons s⁻¹ cm⁻² Å⁻¹]", channel_name=channel.channel_name, full=True)

    return flux_cut, wavelength_cut

def gaussbroad(wavelength, spectra, hwhm):
    #Smooths a spectrum by convolution with a gaussian of specified hwhm.
    # w (input vector) wavelength scale of spectrum to be smoothed
    # s (input vector) spectrum to be smoothed
    # hwhm (input scalar) half width at half maximum of smoothing gaussian.
        #Returns a vector containing the gaussian-smoothed spectrum.
        #Edit History:
        #  -Dec-90 GB,GM Rewrote with fourier convolution algorithm.
        #  -Jul-91 AL	Translated from ANA to IDL.
        #22-Sep-91 JAV	Relaxed constant dispersion check# vectorized, 50% faster.
       #05-Jul-92 JAV	Converted to function, handle nonpositive hwhm.

    #Warn user if hwhm is negative.
    #  if hwhm lt 0.0 then $
    #    message,/info,'Warning! Forcing negative smoothing width to zero.'
        #
        ##Return input argument if half-width is nonpositive.
        #  if hwhm le 0.0 then return,s			#true: no broadening

    #Calculate (uniform) dispersion.

    dw = (wavelength[-1] - wavelength[0]) / len(wavelength)		#wavelength change per pixel

    #gauus=make
    for _ in range(0, len(wavelength)):
        #Make smoothing gaussian# extend to 4 sigma.
        #Note: 4.0 / sqrt(2.0*numpy.log(2.0)) = 3.3972872 & sqrt(numpy.log(2.0))=0.83255461
        #  sqrt(numpy.log(2.0)/pi)=0.46971864 (*1.0000632 to correct for >4 sigma wings)
        if(hwhm > 5*(wavelength[-1] - wavelength[0])): 
            return np.full(len(wavelength),np.sum(spectra)/len(wavelength))
        
        nhalf = int(3.3972872*hwhm/dw)		## points in half gaussian
        ng = 2 * nhalf + 1				## points in gaussian (odd!)
        wg = dw * (np.arange(ng) - (ng-1)/2.0)	#wavelength scale of gaussian
        xg = ( (0.83255461) / hwhm) * wg 		#convenient absisca
        gpro = ( (0.46974832) * dw / hwhm) * np.exp(-xg*xg)#unit area gaussian w/ FWHM
        gpro=gpro/np.sum(gpro)

        # if _ % 1000 == 0:
        #     sigma = float(hwhm) / float(np.sqrt(2.0 * np.log(2.0)))
        #     fwhm = 2.0 * float(hwhm)
        #     half_width = float(nhalf) * float(dw)
            # logging.info("GAUSS PARAMS: wave0=%g wave-1=%g len(wl)=%d hwhm=%g fwhm=%g sigma=%g dw=%g nhalf=%d ng=%d half_width=%g halfwidth_sigma=%g wg0=%g wgmid=%g wglast=%g xg0=%g g0=%g gsum=%g",float(wavelength[0]), float(wavelength[-1]), int(len(wavelength)),float(hwhm), float(fwhm), float(sigma), float(dw), int(nhalf), int(ng), float(half_width), float(half_width / sigma), float(wg[0]), float(wg[len(wg)//2]), float(wg[-1]), float(xg[0]), float(gpro[0]), float(np.sum(gpro)))

    #Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2				## pad pixels on each end
    spad = np.concatenate((np.full(npad,spectra[0]),spectra,np.full(npad,spectra[-1])))
    #Convolve & trim.
    
    sout = np.convolve(spad,gpro,mode='full')			#convolve with gaussian
    sout = sout[npad:npad+len(wavelength)]			#trim to original data/length
    return sout					#return broadened spectrum.

def counts_per_s_px_conv_per_channel(broadened_photon_flux: np.ndarray, wavelength: np.ndarray, channel: SpectroscopyChannel, output_dir, cfg, star: Star):
    """
    Convert photon flux at Earth [photons/cm²/s/Å] into counts/s/pixel for a single channel and gauss broaden it.
    """
    # Get the pixel wavelength grid from the channel and interpolate smoothed Earth flux onto the pixel wavelength grid
    flux_on_pixel = np.interp(channel.wavelength, wavelength, broadened_photon_flux)
    
    logging.info("Channel %s flux_on_pixel sum=%g mean=%g min=%g max=%g", channel.channel_name, flux_on_pixel.sum(), flux_on_pixel.mean(), flux_on_pixel.min(), flux_on_pixel.max())

    # Step 3: convert flux per Angstrom into flux per pixel
    photons_per_pixel_cm2_s = flux_on_pixel * channel.pixel_scale

    # Step 4: apply effective area to get detector counts per second per pixel
    counts_s_px_convolved = photons_per_pixel_cm2_s * channel.effective_area
    logging.info("Channel %s counts_per_s_per_pixel sum=%g mean=%g min=%g max=%g", channel.channel_name, counts_s_px_convolved.sum(), counts_s_px_convolved.mean(), counts_s_px_convolved.min(), counts_s_px_convolved.max())

    if cfg.test_mode:
        dump_1d_for_channel(channel.wavelength, counts_s_px_convolved, output_dir, star.name, DUMP_TAG_COUNTS, channel_name=channel.channel_name, full=True, zoom=True)

    if cfg.produce_Plots:
        plot_1d_for_channel(channel.wavelength, counts_s_px_convolved, output_dir, star, filename_tag=PLOT_TAG_COUNTS, title_text="Convolved Counts", y_label="Counts s⁻¹ pixel⁻¹", channel_name=channel.channel_name, full=True)

    return counts_s_px_convolved
