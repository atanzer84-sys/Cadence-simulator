from dataclasses import dataclass, fields
import numpy as np
import logging

@dataclass(frozen=True, slots=True)
class Channel:
    channel_name: str
    x_pixels: int
    y_pixels: int
    resolution_factor: float
    dark_noise: float
    dark_current_sigma: float
    read_noise: float
    bias_offset: float
    ccd_gain: float
    exposure_s: float
    source_file: str
    effective_area_file: str
    effective_area_wavelength: np.ndarray
    effective_area: np.ndarray
    pixel_scale: float

    
    def __post_init__(self):
        values = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                values.append(f"{f.name}=array(len={len(value)})")
            else:
                values.append(f"{f.name}={value!r}")
        logging.info("Channel created: %s", ", ".join(values))

@dataclass(frozen=True, slots=True)
class SpectroscopyChannel(Channel):
    mode: int
    spread_profile_file: str
    spread_half_height_pix: int

    slit_position_x_arcsec: float
    slit_position_y_arcsec: float
    slope: float
    intercept_pixels: float
    spread_y_positions: np.ndarray | None = None
    spread_y_weights: np.ndarray | None = None
    spread_y_wavelengths: np.ndarray | None = None
    background_type: str | None = None
    background_wavelength: np.ndarray | None = None
    background_flux: np.ndarray | None = None
    sky_pixel_area_arcsec2: float | None = None
    zod_dist: np.ndarray | None = None
    zod_spectrum_wavelength: np.ndarray | None = None
    zod_spectrum_flux: np.ndarray | None = None

@dataclass(frozen=True, slots=True)
class PhotometryChannel(Channel):
    aperture_pix: float
    spread_profile_file: str
    psf_file: str

    psf_radial_distance: np.ndarray | None = None
    psf_radial_flux: np.ndarray | None = None
    spread_positions: np.ndarray | None = None
    spread_y_weights: np.ndarray | None = None
    spread_x_weights: np.ndarray | None = None
    spread_anchors: np.ndarray | None = None

    psf_image: np.ndarray | None = None
    psf_center_x: int | None = None
    psf_center_y: int | None = None

    source_position_x_arcsec: float | None = None
    source_position_y_arcsec: float | None = None