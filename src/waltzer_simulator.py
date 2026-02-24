import sys
import logging
from loaders.load_channel import load_channels_config
from loaders.run_waltzer_context import initialize_waltzer_runtime_context
from loaders.load_stellar_and_planetary_properties import load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from flux.flux_calc import calculateFluxOnEarth
from instrument.detector import counts_per_s_px_conv_all_channels
from frame.frame_pipeline import generate_Frames

def main():
    try:

        run_ctx, user_cfg = initialize_waltzer_runtime_context()
        nuv_channel, vis_channel, _ = load_channels_config(user_cfg)

        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        photon_flux_at_earth_A, wavelengths_total = calculateFluxOnEarth(star, run_ctx)

        # counts per pixel per second convolved to NUV and VIS
        counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis = counts_per_s_px_conv_all_channels(photon_flux_at_earth_A, wavelengths_total, nuv_channel, vis_channel, run_ctx, star)

        # generating bias, dark and science frames for NUV, VIS
        generate_Frames(counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, nuv_channel, vis_channel, run_ctx, star)


    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



