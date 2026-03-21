import pytest
from datetime import datetime

from loaders.run_context import RunContext


@pytest.fixture
def make_run_context(tmp_path):
    def _make_run_context(**overrides):
        base = dict(
            target_name="HD_2685",
            output_dir=tmp_path,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        base.update(overrides)
        return RunContext(**base)

    return _make_run_context
