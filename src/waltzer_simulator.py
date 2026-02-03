from loaders.run_setup import setup_output_directory, setup_logger, load_user_parameters, load_Excel_properties

def main():
    output_dir, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp)

    # Loading User Parameter file
    user_parameters = load_user_parameters()
    print(user_parameters)
    
    # TODO: load CSV, etc.
    planet_param, stellar_param = load_Excel_properties(user_parameters["target_name"])
    print("planet:", planet_param)
    print("star:", stellar_param)

    # TODO: fetch python code from sreejith and integrate it

if __name__ == "__main__":
    main()
