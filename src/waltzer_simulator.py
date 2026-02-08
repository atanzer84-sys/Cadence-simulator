from loaders.run_setup import setup_output_directory, setup_logger, get_repo_root, get_user_parameter_path, load_stellar_and_planetary_properties
from configs.global_config import load_global_config
from configs.user_config import load_user_config, get_user_config
from domain.star import Star
from domain.planet import Planet
from flux.flux_calc import calculateFluxOnEarth
import sys
import logging


def main():
    print("Getting started...")

    output_dir, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp)

    # Loading User Parameter file
    try:
        repo_root = get_repo_root()
        load_global_config(repo_root / "configs" / "global.cfg")
        user_parameter_path = get_user_parameter_path()
        load_user_config(user_parameter_path)
        user_cfg = get_user_config()
        planet_param, stellar_param, required_planet_keys, required_star_keys = load_stellar_and_planetary_properties(user_cfg.target_name)
    except Exception as e:
        logging.exception("Input error while loading user parameters or Excel properties")
        print(f"Input error: {e}")
        sys.exit(1)


    # Create a star and a planet
    star = Star.from_params(stellar_param, required_keys=required_star_keys)
    _ = Planet.from_params(planet_param, required_keys=required_planet_keys)

    # TODO: fetch python code from sreejith and integrate it
    calculateFluxOnEarth(star, output_dir)

if __name__ == "__main__":
    main()
