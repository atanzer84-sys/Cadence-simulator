from loaders.run_setup import setup_output_directory, setup_logger, load_user_parameters, load_excel_properties, get_repo_root
from configs.global_config import load_global_config

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
        user_parameters = load_user_parameters()
        # print(user_parameters)
        planet_param, stellar_param, required_planet_keys, required_star_keys = load_excel_properties(user_parameters["target_name"])
    except Exception as e:
        logging.exception("Input error while loading user parameters or Excel properties")
        print(f"Input error: {e}")
        sys.exit(1)


    # Create a star and a planet
    star = Star.from_params(stellar_param, required_keys=required_star_keys)
    planet = Planet.from_params(planet_param, required_keys=required_planet_keys)

    # TODO: fetch python code from sreejith and integrate it
    calculateFluxOnEarth(star, output_dir)

if __name__ == "__main__":
    main()
