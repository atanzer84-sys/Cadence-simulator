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
    effective_area_file: str
    mode: int
    spread_profile_file: str
    spread_half_height_pix: int
    wavelength: np.ndarray
    effective_area: np.ndarray
    pixel_scale: float
    spread_y_positions: np.ndarray | None = None
    spread_y_weights: np.ndarray | None = None
    spread_y_wavelengths: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PhotometryChannel(Channel):
    pass

