import pytest
from configs.user_config import UserConfig


@pytest.fixture
def make_user_config():
    def _make_user_config(**overrides):
        base = dict(
            target_name="HD 2685",
            total_observation_length_h=20.0,
            exposure_NUV_s=3.0,
            exposure_VIS_s=4.0,
            exposure_IR_s=10.0,
        )
        base.update(overrides)
        return UserConfig(**base)

    return _make_user_config