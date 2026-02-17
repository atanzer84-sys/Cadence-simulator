from dataclasses import dataclass
from loaders.effective_area_loader import load_effective_area_file
import numpy as np
import logging
from configs.global_config import get_global_config
from utils.debug_dumps import dump_1d_array
from utils.plot_spectra import plot_flux_and_photons_windows
from domain.star import Star



@dataclass(frozen=True, slots=True)
class ChannelCalibration:
    name: str
    wavelength: np.ndarray
    effective_area: np.ndarray
    pixel_scale: float

def load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg):

    nuv_wl, nuv_eff, nuv_pixel_scale = load_effective_area_file(nuv_cfg.effective_area_file)
    if len(nuv_wl) != nuv_cfg.x_pixels:
        msg = f"NUV: wavelength grid length {len(nuv_wl)} loaded from effective area file '{nuv_cfg.effective_area_file}' does not match x_pixels={nuv_cfg.x_pixels} from channel cfg '{getattr(nuv_cfg, 'source_file', '')}'"
        logging.error(msg)
        print(msg)
        raise ValueError(msg)

    vis_wl, vis_eff, vis_pixel_scale = load_effective_area_file(vis_cfg.effective_area_file)
    if len(vis_wl) != vis_cfg.x_pixels:
        msg = f"VIS: wavelength grid length {len(vis_wl)} loaded from effective area file '{vis_cfg.effective_area_file}' does not match x_pixels={vis_cfg.x_pixels} from channel cfg '{vis_cfg.source_file}'"
        logging.error(msg)
        print(msg)
        raise ValueError(msg)


    ir_wl, ir_eff, ir_pixel_scale = load_effective_area_file(ir_cfg.effective_area_file)
    if len(ir_wl) != ir_cfg.x_pixels:
        msg = f"IR: wavelength grid length {len(ir_wl)} loaded from effective area file '{ir_cfg.effective_area_file}' does not match x_pixels={ir_cfg.x_pixels} from channel cfg '{ir_cfg.source_file}'"
        logging.error(msg)
        print(msg)
        raise ValueError(msg)

    logging.info("NUV: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(nuv_wl), len(nuv_eff), nuv_pixel_scale)
    logging.info("VIS: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(vis_wl), len(vis_eff), vis_pixel_scale)
    logging.info("IR: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(ir_wl), len(ir_eff), ir_pixel_scale)
    logging.info("Instrument calibration loaded for NUV, VIS, IR")

    nuv_cal = ChannelCalibration(name="NUV", wavelength=nuv_wl, effective_area=nuv_eff, pixel_scale=nuv_pixel_scale)
    vis_cal = ChannelCalibration(name="VIS", wavelength=vis_wl, effective_area=vis_eff, pixel_scale=vis_pixel_scale)
    ir_cal  = ChannelCalibration(name="IR",  wavelength=ir_wl,  effective_area=ir_eff,  pixel_scale=ir_pixel_scale)

    return nuv_cal, vis_cal, ir_cal


def counts_per_s_px_conv_all_channels(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, nuv_cal, vis_cal, ir_cal, output_dir, star: Star):
    logging.info("Starting convolution to instrument")
    print("Starting convolution to instrument")
    cfg = get_global_config()

    counts_s_pixel_convolved_nuv = counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth, wavelengths_total, nuv_cal, output_dir, cfg, star)
    counts_s_pixel_convolved_vis = counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth, wavelengths_total, vis_cal, output_dir, cfg, star)
    counts_s_pixel_convolved_ir = counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth, wavelengths_total, ir_cal, output_dir, cfg, star)

    return counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, counts_s_pixel_convolved_ir


def counts_per_s_px_conv_all_channels_per_channel(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, cal, output_dir, cfg, star: Star):

    """
    Convert photon flux at Earth [photons/cm²/s/Å] into counts/s/pixel for a single channel and gauss broaden it.
    """
    # Step 1: get the pixel wavelength grid from the channel
    # Step 2: interpolate Earth flux onto the pixel wavelength grid
    flux_on_pixel = np.interp(cal.wavelength, wavelengths_total, photon_flux_at_earth)
    logging.info("Channel %s flux_on_pixel sum=%g mean=%g min=%g max=%g", cal.name, flux_on_pixel.sum(), flux_on_pixel.mean(), flux_on_pixel.min(), flux_on_pixel.max())

    # Step 3: convert flux per Angstrom into flux per pixel
    photons_per_pixel_cm2_s = flux_on_pixel * cal.pixel_scale

    # Step 4: apply effective area to get detector counts per second per pixel
    counts_per_s_per_pixel = photons_per_pixel_cm2_s * cal.effective_area
    logging.info("Channel %s counts_per_s_per_pixel sum=%g mean=%g min=%g max=%g", cal.name, counts_per_s_per_pixel.sum(), counts_per_s_per_pixel.mean(), counts_per_s_per_pixel.min(), counts_per_s_per_pixel.max())

    counts_s_pixel_convolved_per_channel =  gaussbroad(cal.wavelength, counts_per_s_per_pixel, cal.pixel_scale)
    logging.info("Channel %s counts_per_s_per_pixel_smoothed sum=%g mean=%g min=%g max=%g", cal.name, counts_s_pixel_convolved_per_channel.sum(), counts_s_pixel_convolved_per_channel.mean(), counts_s_pixel_convolved_per_channel.min(), counts_s_pixel_convolved_per_channel.max())
    

    if cfg.test_mode:
        dump_1d_array(cal.wavelength, counts_s_pixel_convolved_per_channel, output_dir, star.name, f"counts_per_s_per_pixel_smoothed_{cal.name}", full=True, zoom=False)

    if cfg.produce_Plots:
        plot_flux_and_photons_windows(cal.wavelength, counts_s_pixel_convolved_per_channel, output_dir, star, filename_tag=f"counts_per_s_px_{cal.name}", title_text="Convolved Counts", y_label="Counts s⁻¹ pixel⁻¹", cut = False )


    return counts_s_pixel_convolved_per_channel


def gaussbroad(w,s,hwhm):
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
    dw = (w[-1] - w[0]) / len(w)		#wavelength change per pixel
    #gauus=make
    for _ in range(0, len(w)):
        #Make smoothing gaussian# extend to 4 sigma.
        #Note: 4.0 / sqrt(2.0*numpy.log(2.0)) = 3.3972872 & sqrt(numpy.log(2.0))=0.83255461
        #  sqrt(numpy.log(2.0)/pi)=0.46971864 (*1.0000632 to correct for >4 sigma wings)
        if(hwhm > 5*(w[-1] - w[0])): 
            return np.full(len(w),np.sum(s)/len(w))
        nhalf = int(3.3972872*hwhm/dw)		## points in half gaussian
        ng = 2 * nhalf + 1				## points in gaussian (odd!)
        wg = dw * (np.arange(ng) - (ng-1)/2.0)	#wavelength scale of gaussian
        xg = ( (0.83255461) / hwhm) * wg 		#convenient absisca
        gpro = ( (0.46974832) * dw / hwhm) * np.exp(-xg*xg)#unit area gaussian w/ FWHM
        gpro=gpro/np.sum(gpro)

    #Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2				## pad pixels on each end
    spad = np.concatenate((np.full(npad,s[0]),s,np.full(npad,s[-1])))
    #Convolve & trim.
    sout = np.convolve(spad,gpro,mode='full')			#convolve with gaussian
    sout = sout[npad:npad+len(w)]			#trim to original data/length
    return sout					#return broadened spectrum.
