from types import SimpleNamespace
import numpy as np
import pytest
from src.instrument.detector_images import spread_1d_spectrum_to_2d


@pytest.fixture
def channel_cfg_factory():
    def _make(**overrides):
        base = dict(
            x_pixels=10,
            y_pixels=6,
            mode=1,
            dark_noise=0.0,
            bias_offset=0.0,
            read_noise=0.0,
        )
        base.update(overrides)
        return SimpleNamespace(**base)
    return _make

def test_spread_1d_spectrum_to_2d_raises_not_implemented_for_other_mode(channel_cfg_factory):
    cfg = channel_cfg_factory(x_pixels=10, y_pixels=6, mode=2)
    counts = np.zeros(cfg.x_pixels, dtype=np.float64)

    with pytest.raises(NotImplementedError, match=r"mode=2 not implemented yet"):
        spread_1d_spectrum_to_2d(counts, cfg, header=None)


def test_spread_1d_spectrum_to_2d_executes_for_mode_1(channel_cfg_factory):
    cfg = channel_cfg_factory(x_pixels=8, y_pixels=5, mode=1)
    counts = np.ones(cfg.x_pixels, dtype=np.float64)

    image, header = spread_1d_spectrum_to_2d(counts, cfg, header=None)

    assert image.shape == (cfg.y_pixels, cfg.x_pixels)
    assert header is None
    assert np.allclose(image.sum(axis=0), counts)
