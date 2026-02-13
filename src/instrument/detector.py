from dataclasses import dataclass
from loaders.effective_area_loader import load_effective_area_file
import numpy as np
import logging



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

def photons_to_counts_per_pixel_all(photon_flux_at_earth, wavelengths_total, nuv_cal, vis_cal, ir_cal):
    logging.info("Converting Earth flux to counts/s/pixel")
    nuv_counts_per_pixel_s = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, nuv_cal)
    vis_counts_per_pixel_s = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, vis_cal)
    ir_counts_per_pixel_s  = photons_to_counts_per_pixel(photon_flux_at_earth, wavelengths_total, ir_cal)
    return nuv_counts_per_pixel_s, vis_counts_per_pixel_s, ir_counts_per_pixel_s
