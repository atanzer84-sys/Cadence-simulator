import sys
import logging
from pathlib import Path
from typing import Callable, Any
from utils.helpers import ensure_path_under, resolve_path_under
from datetime import datetime
from configs.user_config import load_user_config, get_user_config
from configs.global_config import load_global_config, get_global_config
from utils import debug_dumps
from loaders.run_context import RunContext


def _noop(*args, **kwargs):
    pass


def _select(enabled: bool, fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn if enabled else _noop


def initialize_waltzer_runtime_context():

    print("\n==== LOADING AND INITIALIZING WALTzER SIMULATOR =====")

    output_dir, timestamp_str, timestamp = setup_output_directory()
    setup_logger(output_dir, timestamp_str)
    user_cfg = load_cfg_and_user_config()
    cfg = get_global_config()
    from utils import images
    dump_3d_array = _select(cfg.write_intermediate_arrays, debug_dumps.dump_3d_array)
    dump_1d_array = _select(cfg.write_intermediate_arrays, debug_dumps.dump_1d_array)
    dump_1d_for_channel = _select(cfg.write_intermediate_arrays, debug_dumps.dump_1d_for_channel)
    plot_1d_for_channel = _select(cfg.produce_flux_convolution_plots, images.plot_1d_for_channel)
    plot_flux_and_photons_windows = _select(cfg.produce_flux_convolution_plots, images.plot_flux_and_photons_windows)
    plot_background_star_counts = _select(cfg.produce_background_star_counts_plot, images.plot_background_star_counts)
    write_calibration_frame_png = _select(cfg.write_calibration_frames_png, images.write_calibration_frame_png)
    generate_background_star_visibility_on_science_frame = _select(cfg.write_background_star_science_frames_png, images.generate_background_star_visibility_on_science_frame)

    run_ctx = RunContext(
        target_name=user_cfg.target_name,
        output_dir=output_dir,
        timestamp=timestamp,
        timestamp_str=timestamp_str,
        dump_3d_array=dump_3d_array,
        dump_1d_array=dump_1d_array,
        dump_1d_for_channel=dump_1d_for_channel,
        plot_1d_for_channel=plot_1d_for_channel,
        plot_flux_and_photons_windows=plot_flux_and_photons_windows,
        plot_background_star_counts=plot_background_star_counts,
        write_calibration_frame_png=write_calibration_frame_png,
        generate_background_star_visibility_on_science_frame=generate_background_star_visibility_on_science_frame,
    )
    logging.info("RunContext initialized: target=%s output_dir=%s write_intermediate_arrays=%s produce_flux_convolution_plots=%s write_calibration_frame_png=%s write_background_star_science_frames_png=%s", run_ctx.target_name, run_ctx.output_dir, cfg.write_intermediate_arrays, cfg.produce_flux_convolution_plots, cfg.write_calibration_frames_png, cfg.write_background_star_science_frames_png)
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s", handlers=[logging.FileHandler(log_filename)])

    logging.info("Logger initialized")


def get_repo_root(base_dir: Path | None = None) -> Path:
    p = (base_dir or Path(__file__)).resolve()

    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "input").exists():
            return parent

    raise RuntimeError("Repository root not found")


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
    if len(sys.argv) > 2:
        logging.error("Too many command line arguments: %s", sys.argv)
        print("Usage: python waltzer_simulator.py [parameters_file]")
        sys.exit(1)

    repo_root = get_repo_root()
    if len(sys.argv) == 2:
        parameter_file = ensure_path_under(Path(sys.argv[1]), repo_root)
    else:
        parameter_file = resolve_path_under(repo_root, "input", "parameters.txt")

    msg = f"User parameter file loaded: {parameter_file.resolve()}"
    print(msg)
    logging.info(msg)

    if not parameter_file.exists():
        logging.exception("Parameter file not found: %s", parameter_file)
        print(f"Input error: parameter file not found: {parameter_file}", file=sys.stderr)
        raise SystemExit(1)

    return parameter_file
