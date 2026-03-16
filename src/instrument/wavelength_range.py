import logging
from configs.channel_config import PhotometryChannel, SpectroscopyChannel, Channel

def compute_extended_wavelength_range(channels: list[Channel], margin_A: float = 200.0) -> tuple[float, float]:
    if not channels:
        raise ValueError("At least one channel must be provided when computing wavelength range.")
    wl_min = min(float(c.effective_area_wavelength[0]) for c in channels)
    wl_max = max(float(c.effective_area_wavelength[-1]) for c in channels)

    
    return wl_min - margin_A, wl_max + margin_A


def get_required_wavelength_range(nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None, margin_A: float = 200.0) -> tuple[float, float]:
    channels = [c for c in (nuv, vis, nir) if c is not None]
    wl_min_A, wl_max_A = compute_extended_wavelength_range(channels, margin_A)

    logging.info("WL RANGE: wl_min_A=%g wl_max_A=%g margin_A=%g", wl_min_A, wl_max_A, margin_A)

    return wl_min_A, wl_max_A