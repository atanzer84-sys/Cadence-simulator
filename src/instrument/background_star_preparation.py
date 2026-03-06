from loaders.load_background_stars import lookup_background_stars
from configs.global_config import GlobalConfig
from domain.star_catalog import StarCatalog
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import logging
from configs.channel_config import SpectroscopyChannel, PhotometryChannel, Channel
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel


def populate_background_star_catalog(nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, cfg: GlobalConfig, star: Star):

    background_stars_catalog = lookup_background_stars(ctx, cfg, star)
    calculate_counts_per_px_s(background_stars_catalog, nuv, ctx)
    calculate_counts_per_px_s(background_stars_catalog, vis, ctx)
    calculate_counts_per_px_s(background_stars_catalog, nir, ctx)

    return background_stars_catalog


def calculate_counts_per_px_s(background_stars_catalog: StarCatalog, channel: Channel, ctx: RunContext):
    total = len(background_stars_catalog.stars_by_id)

    for i, (star_id, bg_star) in enumerate(background_stars_catalog.stars_by_id.items(), start=1):

        key = (star_id, channel.channel_name)
        if key in background_stars_catalog.counts_by_id_and_band:
            continue

        logging.info("Calculating Counts / px / s %d/%d for %s channel %s", i, total, star_id, channel.channel_name)
        print(f"Calculating Counts / px / s {i}/{total} for {star_id} channel {channel.channel_name}")

        wavelengths, flux_unred = background_stars_catalog.flux_earth_by_id[star_id]

        counts_s_px = compute_counts_per_s_px_one_channel(flux_unred, wavelengths, channel, ctx, bg_star)

        background_stars_catalog.counts_by_id_and_band[key] = counts_s_px