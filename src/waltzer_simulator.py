from loaders.run_setup import setup_output_directory, setup_logger, load_user_parameters, load_excel_properties
from domain.star import Star
from domain.planet import Planet


def main():
    print("Getting started...")
    
    output_dir, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp)

    # Loading User Parameter file
    user_parameters = load_user_parameters()
    print(user_parameters)
    planet_param, stellar_param, required_planet_keys, required_star_keys = load_excel_properties(user_parameters["target_name"])
    
    # Create a star and a planet
    star = Star.from_params(stellar_param, required_keys=required_star_keys)
    planet = Planet.from_params(planet_param, required_keys=required_planet_keys)
    print("==== STAR ====")
    print(star)
    print("==== PLANET ====")
    print(planet)

    # TODO: fetch python code from sreejith and integrate it

if __name__ == "__main__":
    main()
