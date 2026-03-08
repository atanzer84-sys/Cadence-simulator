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
    plot_background_star_counts: Callable[..., Any]
    write_image_png: Callable[..., Any]
    generate_background_star_visibility_on_science_frame: Callable[..., Any]
