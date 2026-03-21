from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from loaders.run_context import RunContext


# Tests: RunContext
# Behavior: stores all provided values unchanged
def test_run_context_stores_all_fields():
    timestamp = datetime(2025, 1, 1, 12, 0, 0)
    output_dir = Path("/tmp/test-output")

    ctx = RunContext(
        target_name="WASP-69",
        output_dir=output_dir,
        timestamp=timestamp,
    )

    assert ctx.target_name == "WASP-69"
    assert ctx.output_dir == output_dir
    assert ctx.timestamp == timestamp


# Tests: RunContext
# Behavior: rejects field mutation after creation
def test_run_context_is_frozen():
    ctx = RunContext(
        target_name="WASP-69",
        output_dir=Path("/tmp/test-output"),
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )

    with pytest.raises(FrozenInstanceError):
        ctx.target_name = "HD 189733"


# Tests: RunContext
# Behavior: requires all declared fields at construction
def test_run_context_requires_all_fields():
    with pytest.raises(TypeError):
        RunContext(
            target_name="WASP-69",
            output_dir=Path("/tmp/test-output"),
        )
