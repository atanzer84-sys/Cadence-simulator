from loaders.run_setup import *

def main():
    output_dir, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp)

    # Loading User Parameter file
    user_parameters = load_user_parameters()
    print(user_parameters)

    
    # TODO: load CSV, etc.


if __name__ == "__main__":
    main()
