import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
from domain.star_catalog import StarCatalog
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
from utils.images import plot_background_star_counts


def generate_Background_Stars_Image(channel: SpectroscopyChannel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog):

    logging.info("Generating background stars image for channel %s", channel.channel_name)
    print(f"\nGenerating background stars image for channel {channel.channel_name}.")
    nx = channel.x_pixels
    ny = channel.y_pixels
    image = np.zeros((ny, nx), dtype=np.float64)

    background_stars_catalog = calculate_counts_per_px_s(background_stars_catalog, channel, ctx)

    plot_background_star_counts(background_stars_catalog, channel,ctx)

    logging.info("Background stars image done for channel %s sum=%g", channel.channel_name, float(np.sum(image)))


def calculate_counts_per_px_s(background_stars_catalog: StarCatalog, channel: SpectroscopyChannel, ctx: RunContext):
    total = len(background_stars_catalog.stars_by_id)

    for i, (star_id, bg_star) in enumerate(background_stars_catalog.stars_by_id.items(), start=1):
        logging.info("Calculating Counts / px / s %d/%d for %s channel %s", i, total, star_id, channel.channel_name)
        print(f"Calculating Counts / px / s {i}/{total} for {star_id} channel {channel.channel_name}")

        wavelengths, flux_unred = background_stars_catalog.flux_earth_by_id[star_id]

        counts_s_px = compute_counts_per_s_px_one_channel(flux_unred, wavelengths, channel, ctx, bg_star)

        key = (star_id, channel.channel_name)
        background_stars_catalog.counts_by_id_and_band[key] = counts_s_px

    return background_stars_catalog
