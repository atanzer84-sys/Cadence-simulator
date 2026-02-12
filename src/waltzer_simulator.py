from loaders.run_setup import initialize_waltzer_runtime, load_cfg_and_user_config, load_stellar_and_planetary_properties
from domain.star import Star
from domain.planet import Planet
from flux.flux_calc import calculateFluxOnEarth

import sys
import logging


def main():
    try:

        output_dir = initialize_waltzer_runtime()
        user_cfg = load_cfg_and_user_config()

        planet_param, stellar_param, required_planetary_parameters, required_stellar_parameters = load_stellar_and_planetary_properties(user_cfg.target_name)

        star = Star.from_params(stellar_param, required_keys=required_stellar_parameters)
        _ = Planet.from_params(planet_param, required_keys=required_planetary_parameters)

        calculateFluxOnEarth(star, output_dir)



    except Exception as e:
        logging.exception("Fatal error")
        print(f"Input error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
