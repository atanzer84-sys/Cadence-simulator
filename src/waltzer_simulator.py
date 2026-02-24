import sys
import logging
from loaders.load_channel import load_channels_config
from loaders.run_waltzer_context import initialize_waltzer_runtime_context
from loaders.load_stellar_and_planetary_properties import load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from frame.frame_pipeline import generate_Frames
from instrument.prepare_detector_images import prepare_all_detector_images_all_channels
from instrument.build_science_image import build_science_images

def main():
    try:

        run_ctx, user_cfg = initialize_waltzer_runtime_context()
        nuv_channel, vis_channel, _ = load_channels_config(user_cfg)

        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        # calculating flux on earth, convoluting it to instrument properties and returning a 2d image without any additional information
        spectra_2d_nuv, spectra_2d_vis = prepare_all_detector_images_all_channels(star, run_ctx, nuv_channel, vis_channel)

        build_science_images(spectra_2d_nuv, spectra_2d_vis, nuv_channel, vis_channel, run_ctx)
        # # generating bias, dark and science frames for NUV, VIS
        # generate_Frames(spectra_2d_nuv, spectra_2d_vis, nuv_channel, vis_channel, run_ctx, star)


    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



