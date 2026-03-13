from domain.star_catalog import StarCatalog
from loaders.run_waltzer_context import RunContext
import logging
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
import numpy as np
from instrument.wavelength_range import get_required_wavelength_range
from instrument.prepare_detector_images import prepare_star_photon_flux_in_range

def compute_background_stars_counts(background_stars_catalog: StarCatalog, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None, ctx: RunContext) -> StarCatalog:

    wl_min_A, wl_max_A = get_required_wavelength_range(nuv, vis, nir)
    enabled_channels = [c for c in (nuv, vis, nir) if c is not None]
    total = len(background_stars_catalog.stars_by_id)

    if total == 0:
        logging.info("No background stars found for calculation.")
        return background_stars_catalog

    logging.info("Starting Flux Calculation and Detector Convolution for Star in Counts / px / s FOR %d background stars", total)
    print(f"\n==== STARTING FLUX CALCULATION FOR {total} BACKGROUND STARS =====")

    for i, (star_id, bg_star) in enumerate(background_stars_catalog.stars_by_id.items(), start=1):
        if i == 1 or i == total or i % 10 == 0:
            print(f"Flux Calculation and Detector Convolution for Star: {i}/{total} for {star_id}")

        photons_star, wavelengths = prepare_star_photon_flux_in_range(bg_star, ctx, wl_min_A, wl_max_A, announce_user=False)

        for channel in enabled_channels:
            key = (star_id, channel.channel_name)
            if key in background_stars_catalog.counts_by_id_and_band:
                continue

            counts_s_px = compute_counts_per_s_px_one_channel(photons_star, wavelengths, channel, ctx, bg_star)

            if isinstance(channel, SpectroscopyChannel):
                background_stars_catalog.counts_by_id_and_band[key] = counts_s_px.astype(np.float32)
            else:
                background_stars_catalog.counts_by_id_and_band[key] = float(np.sum(counts_s_px))

    return background_stars_catalog  

