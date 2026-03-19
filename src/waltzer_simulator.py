import sys
import logging
from loaders.load_channel import load_channels_config
from loaders.run_waltzer_context import initialize_waltzer_runtime_context
from loaders.load_stellar_and_planetary_properties import load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from instrument.prepare_detector_images import prepare_star_photon_flux_for_channels, prepare_detector_image_spectroscopy, prepare_detector_image_photometry
from instrument.science_image import build_science_images
from instrument.background_star_counts import populate_background_star_counts
from loaders.load_background_stars import lookup_background_stars


def main():
    try:
        
        run_ctx, user_cfg = initialize_waltzer_runtime_context()
        nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg, run_ctx)

        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        # calculating flux on earth once for all enabled channels
        photons_star, wavelengths_total = prepare_star_photon_flux_for_channels(star, run_ctx, nuv_channel, vis_channel, nir_channel)

        # convoluting flux to instrument properties and returning per-channel detector images
        spectra_2d_nuv = prepare_detector_image_spectroscopy(photons_star, wavelengths_total, nuv_channel, run_ctx, star) if nuv_channel is not None else None
        spectra_2d_vis = prepare_detector_image_spectroscopy(photons_star, wavelengths_total, vis_channel, run_ctx, star) if vis_channel is not None else None
        rate_nir = prepare_detector_image_photometry(photons_star, wavelengths_total, nir_channel, run_ctx, star) if nir_channel is not None else None

        # lookup background stars and populate a star catalog with the background stars and convolved counts
        background_stars_catalog = lookup_background_stars(nuv_channel, vis_channel, nir_channel, run_ctx, star)
        background_stars_catalog = populate_background_star_counts(background_stars_catalog, nuv_channel, vis_channel, nir_channel, run_ctx)
        
        # Build and write science frames immediately per channel (streaming)
        build_science_images(spectra_2d_nuv, nuv_channel, run_ctx, star, background_stars_catalog) if nuv_channel is not None else None
        build_science_images(spectra_2d_vis, vis_channel, run_ctx, star, background_stars_catalog) if vis_channel is not None else None
        build_science_images(rate_nir, nir_channel, run_ctx, star, background_stars_catalog) if nir_channel is not None else None        


    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



