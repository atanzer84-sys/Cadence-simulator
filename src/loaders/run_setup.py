import sys
import logging
from loaders.userparameter_loader import load_parameters
from loaders.excel_loader import load_excel_parameters, separate_stellar_planetary_parameters
from loaders.parameter_preprocessing import process_stellar_parameter_values, process_planetary_parameter_values
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

def load_Excel_properties(target_name_user_input):
    try:
        excel_path = _find_excel_file()
        logging.info("Using Excel file '%s' for target '%s'", excel_path, target_name_user_input)
        planet_star_dictionary, target_name = load_excel_parameters(excel_path, target_name_user_input)
        stellar_parameters_excel, planetary_parameters_excel = separate_stellar_planetary_parameters(planet_star_dictionary, target_name)
        planet_param = process_planetary_parameter_values(planetary_parameters_excel)
        stellar_parameters_excel = process_stellar_parameter_values(stellar_parameters_excel)
        return planet_param, stellar_parameters_excel
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Input error: {e}")
        sys.exit(1)

def _find_excel_file(base_dir: Path | None = None):
    repo_root = base_dir or Path(__file__).resolve().parents[2]
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
