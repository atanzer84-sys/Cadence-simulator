"""Pytest conftest: add repo src to path so imports work when run from repo root."""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
