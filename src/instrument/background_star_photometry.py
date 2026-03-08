import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import PhotometryChannel
from domain.star import Star
from domain.star_catalog import StarCatalog


def generate_background_star_photometry_image(channel: PhotometryChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog, roll_angle_deg: float, frame_index: int) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float64)
    background_star_bands: dict[str, dict[str, float]] = {}
    return image, background_star_bands