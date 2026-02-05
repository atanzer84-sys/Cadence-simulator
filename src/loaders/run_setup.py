import sys
import logging
from domain.star import Star
from domain.planet import Planet
from loaders.userparameter_loader import load_parameters
from loaders.excel_loader import load_matching_excel_row_from_excel, load_excel_cfg, map_to_planet_or_star_dictionary
from loaders.parameter_preprocessing import get_missing_properties, clean_and_cast_parameters
from loaders.gaia_lookup import lookup_star_gaia

from pathlib import Path
from datetime import datetime

def setup_output_directory():
    """
    Create a unique output directory under output/.

    Safe for parallel runs: tries to create a directory; if it already exists,
    retries with a different suffix until it succeeds.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)

    output_dir = output_root / timestamp

    # Try base name first, then add _01, _02, ...
    for i in range(0, 10000):
        suffix = "" if i == 0 else f"_{i:02d}"
        output_dir = output_root / f"{timestamp}{suffix}"

        try:
            output_dir.mkdir(parents=True, exist_ok=False)  # atomic
            return output_dir, timestamp
        except FileExistsError:
            # Another process won this name; try the next one.
            continue

    raise RuntimeError("Could not create a unique output directory after many attempts.")

def setup_logger(output_dir, timestamp):
    """Configure logging to a single file in the output directory.

    Logs are written only to the file (no console). The log file is
    named ``waltzer_simulator_<timestamp>.log`` inside ``output_dir``.
    """
    log_filename = output_dir / f"waltzer_simulator_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(log_filename)],
    )

    logging.info("Logger initialized")

def load_user_parameters():
    """Load user parameters from commandline: optional parameter file path.

    Reads ``sys.argv``: at most one optional argument (the parameter file path).
    If no argument is given, uses ``parameters.txt``. Prints usage and exits
    with code 1 if too many arguments are passed, or if the file is missing or
    invalid (ValueError or FileNotFoundError).

    Returns
    -------
    dict
        Parsed parameters (e.g. ``target_name``, ``total_observation_length_h``,
        ``exposure_*``). Only returns on success; exits on error.
    """
    if len(sys.argv) > 2:
        logging.error(
            "Too many command line arguments: %s",
            sys.argv,
        )
        print("Usage: python waltzer_simulator.py [parameters_file]")
        sys.exit(1)

    if len(sys.argv) == 2:
        parameter_file = sys.argv[1]
    else:
        parameter_file = "parameters.txt"

    logging.info("Using parameter file: %s", parameter_file)

    try:
        return load_parameters(parameter_file)
    except ValueError as e:
        logging.exception("Invalid parameter file format: %s", parameter_file)
        print(f"Input error: {e}", file=sys.stderr)
        raise SystemExit(1)

    except FileNotFoundError:
        logging.exception("Parameter file not found: %s", parameter_file)
        print(f"Input error: parameter file not found: {parameter_file}", file=sys.stderr)
        raise SystemExit(1)

def load_excel_properties(target_name_user_input):
    try:
        repo_root = get_repo_root()

        # find the excel file
        excel_path = _find_excel_file(repo_root)
        logging.info("Using Excel file '%s' for target '%s'", excel_path, target_name_user_input)
        
        # load the star's matching row in a mixed dictionary
        planet_star_dictionary, target_name = load_matching_excel_row_from_excel(excel_path, target_name_user_input)

        # load the excel config, because it defines what is a star's and planet's property. easily expandable.
        mapping_path = repo_root / "configs" / "excel_mapping.cfg"
        mapping = load_excel_cfg(mapping_path)

        # getting (still dirty) separate dictionaries for planetary and stellar properties
        planet_params, star_params = map_to_planet_or_star_dictionary(planet_star_dictionary, mapping, target_name)

        # TODO: MISSING PLANET WHEN NEEDED 
        # list all properties that are required by config, empty in excel to prep for gaia lookup
        missing_star = get_missing_properties(star_params, mapping["required_star_parameters"])

        # TODO: GAIA LOOKUP
        if missing_star:
            logging.info("Missing required star keys -> Gaia lookup: %s", missing_star)
            gaia_star_params = lookup_star_gaia(star_params)
            # merge missing only (Excel wins)
            star_params = merge_gaia_into_star_params(star_params, gaia_star_params)

        # now we finally have a list on missing parameters and can throw exceptions, because with missing parameters we can not do our simulation run.
        missing_star_final = get_missing_properties(star_params, mapping["required_star_parameters"])
        logging.info("Missing required star params after enrichment: %s", missing_star_final)

        if missing_star_final:
            raise ValueError(f"Missing required star parameters after GAIA  lookup: {missing_star_final}")

        star_params = clean_and_cast_parameters(star_params, Star)
        planet_params = clean_and_cast_parameters(planet_params, Planet)

        return (
            planet_params,
            star_params,
            mapping["required_planet_parameters"],
            mapping["required_star_parameters"],
        )
    except Exception:
        logging.exception(
            "Failed to load Excel/Gaia properties for target_name_user_input=%r",
            target_name_user_input,
        )
        raise

def get_repo_root(base_dir: Path | None = None) -> Path:
    return base_dir or Path(__file__).resolve().parents[2]

def _find_excel_file(repo_root: Path):
    print("repo_root:", repo_root)
    try:
        # Ignore temporary Excel lock files (e.g. "~$Targets_V10p1.xlsx")
        excel_files = [
            f for f in repo_root.glob("*.xlsx") if not f.name.startswith("~$")
        ]

        if len(excel_files) == 0:
            raise FileNotFoundError(
                f"No Excel file found in repo root ({repo_root})"
            )

        if len(excel_files) > 1:
            names = [f.name for f in excel_files]
            raise ValueError(
                f"Multiple Excel files found in repo root: {names}"
            )

        return excel_files[0]

    except Exception:
        logging.exception(
            "Failed to locate Excel file in repo root: %s",
            repo_root,
        )
        raise

# TODO: NEEDS WORK
def merge_gaia_into_star_params(star_params, gaia_star_params):
    if not gaia_star_params:
        return star_params

    for k, v in gaia_star_params.items():
        current = star_params.get(k)
        if current is None:
            star_params[k] = v
        elif isinstance(current, str) and current.strip() == "":
            star_params[k] = v

    return star_params
