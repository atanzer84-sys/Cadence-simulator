import pytest
import numpy as np
from astropy.io import fits

from frame.frame_class import Frame


@pytest.fixture
def make_header():
    def _make_header(**overrides):
        header = fits.Header()
        for k, v in overrides.items():
            header[k] = v
        return header

    return _make_header


@pytest.fixture
def make_frame(make_header):
    def _make_frame(**overrides):
        base = dict(
            data=np.zeros((10, 10), dtype=float),
            header=make_header(),
            frame_type="science",
            channel_tag="NUV",
        )
        base.update(overrides)
        return Frame(**base)

    return _make_frame