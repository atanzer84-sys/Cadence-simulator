from dataclasses import dataclass
from loaders.effective_area_loader import load_effective_area_file
import numpy as np
import logging
from configs.global_config import get_global_config
from utils.debug_dumps import dump_3d_array, dump_diff_3d_array, dump_1d_array, dump_diff_1d_array



@dataclass(frozen=True, slots=True)
class ChannelCalibration:
    wavelength: np.ndarray
    effective_area: np.ndarray
    pixel_scale: float

def load_instrument_calibration(nuv_cfg, vis_cfg, ir_cfg):

    nuv_wl, nuv_eff, nuv_pixel_scale = load_effective_area_file(nuv_cfg.effective_area_file)

    vis_wl, vis_eff, vis_pixel_scale = load_effective_area_file(vis_cfg.effective_area_file)

    ir_wl, ir_eff, ir_pixel_scale = load_effective_area_file(ir_cfg.effective_area_file)

    logging.info("NUV: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(nuv_wl), len(nuv_eff), nuv_pixel_scale)
    logging.info("VIS: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(vis_wl), len(vis_eff), vis_pixel_scale)
    logging.info("IR: rows(WL)=%d, rows(EA)=%d, pixel_scale=%s", len(ir_wl), len(ir_eff), ir_pixel_scale)
    logging.info("Instrument calibration loaded for NUV, VIS, IR")

    nuv_cal = ChannelCalibration(wavelength=nuv_wl, effective_area=nuv_eff, pixel_scale=nuv_pixel_scale)
    vis_cal = ChannelCalibration(wavelength=vis_wl, effective_area=vis_eff, pixel_scale=vis_pixel_scale)
    ir_cal = ChannelCalibration(wavelength=ir_wl, effective_area=ir_eff, pixel_scale=ir_pixel_scale)

    return nuv_cal, vis_cal, ir_cal


def convoluteToInstrument(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, nuv_cal, vis_cal, ir_cal, output_dir):
    logging.info("Starting convolution to instrument response")
    print("Starting convolution to instrument response")

    cfg = get_global_config()

    nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel = photons_to_counts_per_pixel_all(
        photon_flux_at_earth, wavelengths_total, nuv_cal, vis_cal, ir_cal
    )
    if cfg.test_mode:
        dump_1d_array(nuv_cal.wavelength, nuv_counts_s_pixel, output_dir, "", "photons_to_counts_per_pixel_NUV", full = True, zoom = False)
        dump_1d_array(vis_cal.wavelength, vis_counts_s_pixel, output_dir, "", "photons_to_counts_per_pixel_VIS", full = True, zoom = False)
        dump_1d_array(ir_cal.wavelength, ir_counts_s_pixel, output_dir, "", "photons_to_counts_per_pixel_IR", full = True, zoom = False)

    logging.info("Counts per second per pixel computed for all channels")

    nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel = gauss_broad_all_channels(
        nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel, nuv_cal, vis_cal, ir_cal
    )
    if cfg.test_mode:
        dump_1d_array(nuv_cal.wavelength, nuv_counts_s_pixel, output_dir, "", "gauss_broad_NUV", full = True, zoom = False)
        dump_1d_array(vis_cal.wavelength, vis_counts_s_pixel, output_dir, "", "gauss_broad_VIS", full = True, zoom = False)
        dump_1d_array(ir_cal.wavelength, ir_counts_s_pixel, output_dir, "", "gauss_broad_IR", full = True, zoom = False)

    logging.info("Instrument convolution complete")

    return nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel


def photons_to_counts_per_pixel_all(photon_flux_at_earth, wavelengths_total, nuv_cal, vis_cal, ir_cal):
    logging.info("Converting Earth flux to counts/s/pixel")
    nuv_counts_per_pixel_s = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, nuv_cal)
    vis_counts_per_pixel_s = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, vis_cal)
    ir_counts_per_pixel_s  = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, ir_cal)
    return nuv_counts_per_pixel_s, vis_counts_per_pixel_s, ir_counts_per_pixel_s

def photons_to_counts_per_pixel(photon_flux_at_earth: np.ndarray, wavelengths_total: np.ndarray, cal: ChannelCalibration) -> np.ndarray:
    """
    Convert photon flux at Earth [photons/cm²/s/Å] into counts/s/pixel for a single channel.
    """

    # Step 1: get the pixel wavelength grid from the channel
    pixel_wavelengths = cal.wavelength  # 1D array, length = number of pixels

    # Step 2: interpolate Earth flux onto the pixel wavelength grid
    flux_on_pixel = np.interp(pixel_wavelengths, wavelengths_total, photon_flux_at_earth)

    # Step 3: convert flux per Angstrom into flux per pixel
    photons_per_pixel_cm2_s = flux_on_pixel * cal.pixel_scale

    # Step 4: apply effective area to get detector counts per second per pixel
    counts_per_s_per_pixel = photons_per_pixel_cm2_s * cal.effective_area

    # Step 5: return the counts per pixel array
    return counts_per_s_per_pixel

def gauss_broad_all_channels(nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel, nuv_cal, vis_cal, ir_cal):
    logging.info("Applying instrumental Gaussian broadening to all channels")

    logging.info("NUV: Applying Gaussian broadening (HWHM=%s Å)", nuv_cal.pixel_scale)
    nuv_counts_s_pixel = gaussbroad(nuv_cal.wavelength, nuv_counts_s_pixel, nuv_cal.pixel_scale)

    logging.info("VIS: Applying Gaussian broadening (HWHM=%s Å)", vis_cal.pixel_scale)
    vis_counts_s_pixel = gaussbroad(vis_cal.wavelength, vis_counts_s_pixel, vis_cal.pixel_scale)

    logging.info("IR: Applying Gaussian broadening (HWHM=%s Å)", ir_cal.pixel_scale)
    ir_counts_s_pixel  = gaussbroad(ir_cal.wavelength,  ir_counts_s_pixel,  ir_cal.pixel_scale)

    logging.info("Gaussian broadening complete for all channels")

    return nuv_counts_s_pixel, vis_counts_s_pixel, ir_counts_s_pixel


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
    for i in range(0, len(w)):
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
