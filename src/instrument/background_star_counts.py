import logging
import numpy as np
from domain.star_catalog import StarCatalog
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
from instrument.prepare_detector_images import calculate_photon_flux_density_on_Earth
from instrument.background_star_spectroscopy import spectroscopy_radius_arcsec

def populate_background_star_counts(background_stars_catalog: StarCatalog, nuv: SpectroscopyChannel | None, vis: SpectroscopyChannel | None, nir: PhotometryChannel | None, ctx: RunContext) -> StarCatalog:

    enabled_channels = [c for c in (nuv, vis, nir) if c is not None]
    total = len(background_stars_catalog.stars_by_id)

    if total == 0:
        logging.info("No background stars found for calculation.")
        return background_stars_catalog

    logging.info("Starting Flux Calculation and Detector Convolution for Star in Counts / px / s FOR %d background stars", total)
    print(f"\n==== STARTING FLUX CALCULATION FOR {total} BACKGROUND STARS =====")

    for i, (star_id, bg_star) in enumerate(background_stars_catalog.stars_by_id.items(), start=1):
        if i == 1 or i == total or i % 10 == 0:
            print(f"Flux Calculation and Detector Convolution for Star: {i}/{total}")

        separation_arcsec = background_stars_catalog.get_separation_arcsec(star_id)

        photons_star, wavelengths = calculate_photon_flux_density_on_Earth(bg_star, ctx, nuv, vis, nir, announce_user=False, background_star=True)

        for channel in enabled_channels:
            key = (star_id, channel.channel_name)
            if key in background_stars_catalog.counts_by_id_and_band:
                logging.info("BG STAR skip existing counts: star_id=%s channel=%s", star_id, channel.channel_name)
                continue

            if isinstance(channel, SpectroscopyChannel):
                slit_radius_arcsec = spectroscopy_radius_arcsec(channel)
                if separation_arcsec > slit_radius_arcsec:
                    logging.info("BG STAR skip compute_counts_per_s_px_one_channel: star_id=%s channel=%s sep_arcsec=%.3f slit_radius_arcsec=%.3f", star_id, channel.channel_name, separation_arcsec, slit_radius_arcsec)
                    continue

                counts_s_px = compute_counts_per_s_px_one_channel(photons_star, wavelengths, channel, ctx, bg_star, background_star=True)
                background_stars_catalog.counts_by_id_and_band[key] = counts_s_px.astype(np.float32)

            else:
                counts_s_px = compute_counts_per_s_px_one_channel(photons_star, wavelengths, channel, ctx, bg_star, background_star=True)
                background_stars_catalog.counts_by_id_and_band[key] = float(np.sum(counts_s_px))

    return background_stars_catalog