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
    """Create a timestamped output directory under ``output/``.

    Creates ``output/<YYYYMMDD_HHMMSS>``. If that path already exists,
    uses ``output/<YYYYMMDD_HHMMSS>_01``, ``_02``, etc., until a new
    directory is created.

    Returns
    -------
    tuple[Path, str]
        ``(output_dir, timestamp)`` where ``output_dir`` is the created
        directory and ``timestamp`` is the string ``YYYYMMDD_HHMMSS``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)

    output_dir = output_root / timestamp

    if output_dir.exists():
        i = 1
        while True:
            candidate = output_root / f"{timestamp}_{i:02d}"
            if not candidate.exists():
                output_dir = candidate
                break
            i += 1

    output_dir.mkdir(parents=True, exist_ok=False)

    return output_dir, timestamp

def setup_logger(output_dir, timestamp):
    """Configure logging to a single file in the output directory.

    Logs are written only to the file (no console). The log file is
    named ``waltzer_simulator_<timestamp>.log`` inside ``output_dir``.
    """
    log_filename = output_dir / f"waltzer_simulator_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
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
        print("Usage: python waltzer_simulator.py [parameters_file]")
        sys.exit(1)

    if len(sys.argv) == 2:
        parameter_file = sys.argv[1]
    else:
        parameter_file = "parameters.txt"

    try:
        return load_parameters(parameter_file)
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Input error: parameter file not found: {parameter_file}")
        sys.exit(1)

def load_excel_properties(target_name_user_input):
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

def get_repo_root(base_dir: Path | None = None) -> Path:
    return base_dir or Path(__file__).resolve().parents[2]

def _find_excel_file(repo_root: Path):
    # Ignore temporary Excel lock files (e.g. \"~$Targets_V10p1.xlsx\" created while the workbook is open in Excel).
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
