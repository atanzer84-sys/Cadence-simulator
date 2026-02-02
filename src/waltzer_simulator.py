import sys
import logging
from loaders.input_loader import load_parameters
from pathlib import Path
from datetime import datetime


def load_user_parameters():
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

def setup_logger(output_dir, timestamp):
    log_filename = output_dir / f"waltzer_simulator_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename)],
    )

    logging.info("Logger initialized")


def setup_output_directory():
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

def main():
    output_dir, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp)

    params = load_user_parameters()
    print(params)
    # TODO: load CSV, etc.


if __name__ == "__main__":
    main()
