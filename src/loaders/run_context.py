from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any


@dataclass(frozen=True)
class RunContext:
    target_name: str
    output_dir: Path
    timestamp: datetime
    timestamp_str: str
    dump_3d_array: Callable[..., Any]
    dump_1d_array: Callable[..., Any]
    dump_1d_for_channel: Callable[..., Any]
    plot_1d_for_channel: Callable[..., Any]
    plot_flux_and_photons_windows: Callable[..., Any]
    plot_star_counts_vs_noise_spectroscopy: Callable[..., Any]
    plot_star_counts_vs_noise_photometry: Callable[..., Any]
    write_calibration_frame_png: Callable[..., Any]
    write_science_frame_png: Callable[..., Any]
    write_science_frame_component_png: Callable[..., Any]
    generate_background_star_visibility_on_science_frame: Callable[..., Any]
