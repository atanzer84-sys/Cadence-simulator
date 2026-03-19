import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"

if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tests.fixtures.channel_fixture import *
from tests.fixtures.frame_fixture import *
from tests.fixtures.global_config_fixture import *
from tests.fixtures.planet_fixture import *
from tests.fixtures.run_context_fixture import *
from tests.fixtures.star_catalog_fixture import *
from tests.fixtures.star_fixture import *
from tests.fixtures.user_config_fixture import *
from datetime import datetime

@pytest.fixture
def fixed_timestamp():
    return datetime(2024, 1, 1, 12, 0, 0)