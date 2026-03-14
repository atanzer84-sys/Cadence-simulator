"""Shared minimal RunContext-like object for tests that need only target_name and output_dir."""

from pathlib import Path
from types import SimpleNamespace

BASE_RUN_CONTEXT = {
    "target_name": "HD_2685",
}


def run_context(output_dir, **overrides):
    """Minimal RunContext-like object. Pass output_dir (Path or str); override target_name etc. as needed."""
    d = dict(BASE_RUN_CONTEXT)
    d["output_dir"] = Path(output_dir)
    d.update(overrides)
    return SimpleNamespace(**d)
