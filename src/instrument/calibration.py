from dataclasses import dataclass
import numpy as np

from loaders.effective_area_loader import load_effective_area_file
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

    nuv_cal = ChannelCalibration(wavelength=nuv_wl, effective_area=nuv_eff, pixel_scale=nuv_pixel_scale)
    vis_cal = ChannelCalibration(wavelength=vis_wl, effective_area=vis_eff, pixel_scale=vis_pixel_scale)
    ir_cal = ChannelCalibration(wavelength=ir_wl, effective_area=ir_eff, pixel_scale=ir_pixel_scale)

    return nuv_cal, vis_cal, ir_cal
