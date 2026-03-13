from loaders.load_background_stars import lookup_background_stars
from domain.star_catalog import StarCatalog
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import logging
from configs.channel_config import SpectroscopyChannel, PhotometryChannel, Channel
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
import numpy as np
from configs.global_config import get_global_config


# TODO:
def populate_background_star_catalog(nuv: SpectroscopyChannel, vis: SpectroscopyChannel, nir: PhotometryChannel, ctx: RunContext, star: Star):

    cfg = get_global_config()

    wl_min_A = min(float(nuv.effective_area_wavelength[0]), float(vis.effective_area_wavelength[0]), float(nir.effective_area_wavelength[0]))
    wl_max_A = max(float(nuv.effective_area_wavelength[-1]), float(vis.effective_area_wavelength[-1]), float(nir.effective_area_wavelength[-1]))

    background_stars_catalog = lookup_background_stars(ctx, cfg, star, wl_min_A, wl_max_A)
    calculate_counts_per_px_s(background_stars_catalog, nuv, ctx)
    calculate_counts_per_px_s(background_stars_catalog, vis, ctx)
    calculate_counts_per_px_s(background_stars_catalog, nir, ctx)

    return background_stars_catalog


def calculate_counts_per_px_s(background_stars_catalog: StarCatalog, channel: Channel, ctx: RunContext):
    total = len(background_stars_catalog.stars_by_id)
    logging.info("Calculating counts for %d background stars in channel %s", total, channel.channel_name)
    
    for i, (star_id, bg_star) in enumerate(background_stars_catalog.stars_by_id.items(), start=1):

        key = (star_id, channel.channel_name)
        if key in background_stars_catalog.counts_by_id_and_band:
            continue

        print(f"Calculating Counts / px / s {i}/{total} for {star_id} channel {channel.channel_name}")

        wavelengths, flux_unred = background_stars_catalog.flux_earth_by_id[star_id]

        counts_s_px = compute_counts_per_s_px_one_channel(flux_unred, wavelengths, channel, ctx, bg_star)

        if isinstance(channel, SpectroscopyChannel):
            # store full 1D array (float32)
            background_stars_catalog.counts_by_id_and_band[key] = counts_s_px.astype(np.float32)
        else:
            # store only scalar total flux
            background_stars_catalog.counts_by_id_and_band[key] = float(np.sum(counts_s_px))