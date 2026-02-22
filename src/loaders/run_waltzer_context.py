import functools
import inspect
import sys
import logging
from pathlib import Path
from datetime import datetime

from configs.user_config import load_user_config, get_user_config
from configs.global_config import load_global_config, get_global_config
from utils import debug_dumps

from dataclasses import dataclass


def _noop(f):
    @functools.wraps(f)
    def noop(*args, **kwargs):
        pass
    noop.__signature__ = inspect.signature(f)
    return noop


class _NoOpTestMode:
    dump_3d_array = staticmethod(_noop(debug_dumps.dump_3d_array))
    dump_1d_array = staticmethod(_noop(debug_dumps.dump_1d_array))
    dump_1d_for_channel = staticmethod(_noop(debug_dumps.dump_1d_for_channel))


class _RealTestMode:
    dump_3d_array = staticmethod(debug_dumps.dump_3d_array)
    dump_1d_array = staticmethod(debug_dumps.dump_1d_array)
    dump_1d_for_channel = staticmethod(debug_dumps.dump_1d_for_channel)


_NOOP = _NoOpTestMode()


class _NoOpProducePlots:
    @staticmethod
    def plot_1d_for_channel(*args, **kwargs):
        pass

    @staticmethod
    def plot_flux_and_photons_windows(*args, **kwargs):
        pass


_NOOP_PLOTS = _NoOpProducePlots()


def _create_produce_plots():
    """Import images lazily to avoid circular import (images imports RunContext)."""
    from utils import images

    class _RealProducePlots:
        plot_1d_for_channel = staticmethod(images.plot_1d_for_channel)
        plot_flux_and_photons_windows = staticmethod(images.plot_flux_and_photons_windows)

    return _RealProducePlots() if get_global_config().produce_Plots else _NOOP_PLOTS


@dataclass(frozen=True)
class RunContext:
    target_name: str
    output_dir: Path
    timestamp: datetime
    timestamp_str: str
    test_mode: _NoOpTestMode | _RealTestMode
    produce_plots: _NoOpProducePlots


def initialize_waltzer_runtime_context():
    print("Getting started...")
    output_dir, timestamp_str, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp_str)
    user_cfg = load_cfg_and_user_config()
    test_mode = _RealTestMode() if get_global_config().test_mode else _NOOP
    produce_plots = _create_produce_plots()

    run_ctx = RunContext(
        target_name=user_cfg.target_name,
        output_dir=output_dir,
        timestamp=timestamp,
        timestamp_str=timestamp_str,
        test_mode=test_mode,
        produce_plots=produce_plots,
    )
    logging.info("RunContext initialized: %s", run_ctx)

    return run_ctx, user_cfg

def load_cfg_and_user_config():
    # load global config (cached, once per run)
    repo_root = get_repo_root()
    load_global_config(repo_root / "configs" / "global.cfg")

    # load user config, not cached, pass it through
    user_parameter_path = get_user_parameter_path()
    load_user_config(user_parameter_path)
    user_cfg = get_user_config()

    return user_cfg

def setup_output_directory():
    """
    Create a unique output directory under output/.

    Safe for parallel runs: tries to create a directory; if it already exists,
    retries with a different suffix until it succeeds.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_time = datetime.now()

    output_root = get_repo_root() / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / timestamp_str

    # Try base name first, then add _01, _02, ...
    for i in range(0, 10000):
        suffix = "" if i == 0 else f"_{i:02d}"
        output_dir = output_root / f"{timestamp_str}{suffix}"

        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            print(f"Output directory created at: {output_dir.resolve()}")
            return output_dir, timestamp_str, base_time
        except FileExistsError:
            # Another process won this name; try the next one.
            continue

    raise RuntimeError("Could not create a unique output directory after many attempts.")

def setup_logger(output_dir, timestamp_str):
    """Configure logging to a single file in the output directory.

    Logs are written only to the file (no console). The log file is
    named ``waltzer_simulator_<timestamp>.log`` inside ``output_dir``.
    """
    filename = f"waltzer_simulator_{timestamp_str}.log"

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
        parameter_file = get_repo_root() / "input" / "parameters.txt"

    logging.info("Using parameter file: %s", parameter_file.resolve())
    print("User parameter file loaded: ", parameter_file.resolve())

    # Validate existence
    if not parameter_file.exists():
        logging.exception("Parameter file not found: %s", parameter_file)
        print(f"Input error: parameter file not found: {parameter_file}", file=sys.stderr)
        raise SystemExit(1)

    return parameter_file
