from loaders.run_setup import initialize_waltzer_runtime, load_cfg_and_user_config, load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from flux.flux_calc import calculateFluxOnEarth
from instrument.detector import load_channel_response_from_effective_area, counts_per_s_px_conv_all_channels
from instrument.detector_images import generate_bias_dark_frames
import sys
import logging

def main():
    try:

        output_dir = initialize_waltzer_runtime()
        user_cfg, nuv_cfg, vis_cfg, ir_cfg = load_cfg_and_user_config()
        nuv_cal, vis_cal, ir_cal = load_channel_response_from_effective_area(nuv_cfg, vis_cfg, ir_cfg)
        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        photon_flux_at_earth_A, wavelengths_total = calculateFluxOnEarth(star, output_dir)

        counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, counts_s_pixel_convolved_ir = counts_per_s_px_conv_all_channels(photon_flux_at_earth_A, wavelengths_total, nuv_cal, vis_cal, ir_cal, output_dir, star)

        generate_bias_dark_frames(nuv_cfg, vis_cfg, user_cfg, output_dir, star)


    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



