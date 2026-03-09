import sys
import logging
from loaders.load_channel import load_channels_config
from loaders.run_waltzer_context import initialize_waltzer_runtime_context
from loaders.load_stellar_and_planetary_properties import load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from frame.frame_pipeline import generate_frames
from instrument.prepare_detector_images import prepare_all_detector_images_all_channels
from instrument.build_science_image import build_science_images

from memory_profiler import profile

@profile
def main():
    try:
        
        run_ctx, user_cfg = initialize_waltzer_runtime_context()
        nuv_channel, vis_channel, nir_channel = load_channels_config(user_cfg, run_ctx)

        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        # calculating flux on earth, convoluting it to instrument properties and returning a 2d image without any additional information
        spectra_2d_nuv, spectra_2d_vis, rate_nir = prepare_all_detector_images_all_channels(star, run_ctx, nuv_channel, vis_channel, nir_channel)

        # generating single science images that are then stacked
        nuv_images, vis_images, nir_images = build_science_images(spectra_2d_nuv, spectra_2d_vis, rate_nir, nuv_channel, vis_channel, nir_channel, run_ctx, star)
        
        # generating bias, dark and science frames (fits) for NUV, VIS
        generate_frames(nuv_images, vis_images, nir_images, nuv_channel, vis_channel, nir_channel, run_ctx, star)


    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



