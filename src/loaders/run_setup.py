import sys
import logging
import numpy as np
from domain.star import Star
from domain.planet import Planet
from configs.global_config import get_global_config
from loaders.excel_loader import load_matching_excel_row_from_excel, load_excel_cfg, map_to_planet_or_star_dictionary
from loaders.parameter_preprocessing import get_missing_properties, clean_and_cast_parameters
from loaders.gaia_lookup import lookup_star_gaia
from astropy.io import ascii
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
            output_dir.mkdir(parents=True, exist_ok=False)
            print(f"Output directory created at: {output_dir.resolve()}")
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
    filename = f"waltzer_simulator_{timestamp}.log"
    log_filename = output_dir / filename
    print(f"Log file created at: {log_filename.resolve()}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(log_filename)],
    )

    logging.info("Logger initialized")

def get_repo_root(base_dir: Path | None = None) -> Path:
    return base_dir or Path(__file__).resolve().parents[2]

def get_user_parameter_path():
    """Determine which user parameter file to use based on CLI arguments.

    Reads ``sys.argv``: at most one optional argument (the parameter file path).
    If no argument is given, uses ``parameters.txt``. Prints usage and exits
    with code 1 if too many arguments are passed, or if the file is missing.

    Returns
    -------
    pathlib.Path
        Path to the parameter file. Only returns on success; exits on error.
    """
    # Too many arguments
    if len(sys.argv) > 2:
        logging.error("Too many command line arguments: %s", sys.argv)
        print("Usage: python waltzer_simulator.py [parameters_file]")
        sys.exit(1)

    # One argument → use it
    if len(sys.argv) == 2:
        parameter_file = Path(sys.argv[1])
    else:
        # No argument → default
        parameter_file = Path("input/parameters.txt")

    logging.info("Using parameter file: %s", parameter_file.resolve())
    print("User parameter file loaded: ", parameter_file.resolve())

    # Validate existence
    if not parameter_file.exists():
        logging.exception("Parameter file not found: %s", parameter_file)
        print(f"Input error: parameter file not found: {parameter_file}", file=sys.stderr)
        raise SystemExit(1)

    return parameter_file

def load_stellar_and_planetary_properties(target_name_user_input):
    try:
        repo_root = get_repo_root()

        # find the excel file
        excel_path = _find_excel_file(repo_root / "input")
        
        # load the star's matching row in a mixed dictionary
        planet_star_dictionary, target_name = load_matching_excel_row_from_excel(excel_path, target_name_user_input)

        # load the excel config, because it defines what is a star's and planet's property. easily expandable.
        mapping_path = repo_root / "configs" / "excel_mapping.cfg"
        mapping = load_excel_cfg(mapping_path)

        # getting (still dirty) separate dictionaries for planetary and stellar properties
        planet_params, star_params = map_to_planet_or_star_dictionary(planet_star_dictionary, mapping, target_name)

        # TODO: MISSING PLANET WHEN NEEDED 
        # list all properties that are required by config, empty in excel to prep for gaia lookup
        missing_star = get_missing_properties(star_params, mapping["required_stellar_parameters"])
        
        if missing_star:
            gaia_star_params = lookup_star_gaia(star_params, missing_star)
            # merge missing only (Excel wins)
            star_params = merge_gaia_into_star_params(star_params, gaia_star_params)

        # getting spectral type from mamjeck table.
        mamajek_path = repo_root / "data" / "stellar_param_mamjeck.txt"
        star_params = infer_mamajek_spectral_type(star_params, mamajek_path)
        star_params = apply_log_r_fallback(star_params)

        # now we finally have a list on missing parameters and can throw exceptions, because with missing parameters we can not do our simulation run.
        missing_star_final = get_missing_properties(star_params, mapping["required_stellar_parameters"])

        if missing_star_final:
            raise ValueError(f"Missing required star parameters after GAIA lookup: {missing_star_final}")

        star_params = clean_and_cast_parameters(star_params, Star)
        planet_params = clean_and_cast_parameters(planet_params, Planet)

        print("Excel file loaded, parsed and cleaned.")
        return (
            planet_params,
            star_params,
            mapping["required_planetary_parameters"],
            mapping["required_stellar_parameters"],
        )
    except Exception:
        logging.exception(
            "Failed to load all required properties after Excel/Mamjeck/Gaia lookup for target_name_user_input=%r",
            target_name_user_input,
        )
        raise

def _find_excel_file(repo_root: Path):
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

        logging.info("Using Excel file to look up target: '%s'", excel_files[0])

        return excel_files[0]

    except Exception:
        logging.exception(
            "Failed to locate Excel file in repo root: %s",
            repo_root,
        )
        raise

def merge_gaia_into_star_params(star_params, gaia_star_params):
    if not gaia_star_params:
        logging.info("No Gaia parameters to merge.")
        return star_params

    for k, v in gaia_star_params.items():
        current = star_params.get(k)

        if current is None or (isinstance(current, str) and current.strip() == ""):
            logging.info("Gaia merge: setting %s = %r", k, v)
            star_params[k] = v
        else:
            logging.info("Gaia merge: keeping existing %s = %r", k, current)

    return star_params

def infer_mamajek_spectral_type(star_params, mamajek_path):
    mamajek_path = str(mamajek_path)
    logging.info("Loading Mamajek table from %s", mamajek_path)

    data = ascii.read(mamajek_path, comment="#")
    logging.info("Mamajek table loaded successfully (%d rows)", len(data))
    Sp = np.array(data["col1"])
    T_book = np.array(data["col2"], dtype=float)

    Teff_raw = star_params.get("effective_temperature")

    if Teff_raw is None:
        logging.error(
            "Mamajek inference aborted: 'effective_temperature' is missing. Current star_params keys: %s", list(star_params.keys()),
        )
        raise ValueError(
                "Cannot infer spectral type: effective_temperature is None. "
            )
    try:
        Teff = float(Teff_raw)

    except (TypeError, ValueError) as e:
        logging.error(
            "Mamajek inference failed: effective_temperature=%r is not convertible to float.",
            Teff_raw,
        )
        raise ValueError(
            f"Cannot infer spectral type: invalid effective_temperature value {Teff_raw!r}"
        ) from e

    idx = int(np.abs(T_book - Teff).argmin())
    spectral_type = str(Sp[idx])

    star_params["spectral_type"] = spectral_type

    logging.info("Set spectral_type=%s inferred from Teff=%s K",spectral_type,Teff)

    return star_params

def apply_log_r_fallback(star_params: dict) -> dict:
    global_config = get_global_config()
    if not global_config.enable_log_r_fallback:
        logging.info("log_r fallback skipped: enable_log_r_fallback=%s", global_config.enable_log_r_fallback)
        return star_params

    if star_params.get("log_r") is not None:
        return star_params

    Teff = star_params.get("effective_temperature")
    if Teff is None:
        logging.info("log_r fallback skipped: effective_temperature missing.")
        return star_params

    try:
        Teff = float(Teff)
    except Exception:
        logging.exception("log_r fallback failed: invalid Teff=%r", Teff)
        return star_params

    threshold = global_config.log_r_teff_threshold
    hot_val = global_config.log_r_hot_value
    cool_val = global_config.log_r_cool_value

    if Teff > threshold:
        star_params["log_r"] = hot_val
    else:
        star_params["log_r"] = cool_val

    logging.info("log_r fallback applied: Teff=%s threshold=%s -> log_r=%s", Teff, threshold, star_params["log_r"])

    return star_params
