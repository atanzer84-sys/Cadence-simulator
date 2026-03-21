from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    target_name: str
    output_dir: Path
    timestamp: datetime
