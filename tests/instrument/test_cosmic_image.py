import numpy as np
from types import SimpleNamespace

from configs.global_config import GlobalConfig
from instrument.cosmic_image import generate_cosmic_rays


# Minimal channel-like object used for cosmic-ray image generation tests.
def _channel(x_pixels: int = 8, y_pixels: int = 6):
    return SimpleNamespace(
        channel_name="TEST",
        x_pixels=x_pixels,
        y_pixels=y_pixels,
    )


class _StubWriteImagePng:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, str, object]] = []

    def write_image(self, array, frame_type: str, ctx, channel, show_stats: bool = True):
        self.calls.append((array, frame_type, channel))


def _ctx(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        write_image_png=_StubWriteImagePng(),
    )


_COMMON_CFG = dict(
    line_core_emission=False,
    interstellar_absorption=False,
    mg2_col=None,
    mg1_col=None,
    fe2_col=None,
    sigmaMg22=0.257,
    sigmaMg21=0.288,
    enable_log_r_fallback=False,
    log_r_teff_threshold=0.0,
    log_r_hot_value=0.0,
    log_r_cool_value=0.0,
    n_non_science_frames=0,
    write_non_science_frames_png=False,
    n_science_frames_per_channel=1,
    write_science_frames_png=False,
    magnitude_cutoff=20.0,
    test_mode=True,
    produce_Plots=False,
)


def _cfg_zero_rays() -> GlobalConfig:
    params = dict(_COMMON_CFG)
    params.update(
        cosmic_rays_min=0,
        cosmic_rays_max=0,
        cosmic_ray_signal_electrons=72000,
        cosmic_ray_length_min_px=1,
        cosmic_ray_length_max_px=3,
    )
    return GlobalConfig(**params)


def _cfg_three_rays() -> GlobalConfig:
    params = dict(_COMMON_CFG)
    params.update(
        cosmic_rays_min=3,
        cosmic_rays_max=3,
        cosmic_ray_signal_electrons=50,
        cosmic_ray_length_min_px=2,
        cosmic_ray_length_max_px=4,
    )
    return GlobalConfig(**params)


def _cfg_single_ray_small_detector() -> GlobalConfig:
    params = dict(_COMMON_CFG)
    params.update(
        cosmic_rays_min=1,
        cosmic_rays_max=1,
        cosmic_ray_signal_electrons=123,
        cosmic_ray_length_min_px=5,
        cosmic_ray_length_max_px=5,
    )
    return GlobalConfig(**params)


# No cosmic rays configured should yield an all-zero image but still trigger one COSMIC_ONLY write.
def test_generate_cosmic_rays_zero_rays(tmp_path):
    ch = _channel()
    ctx = _ctx(tmp_path)
    cfg = _cfg_zero_rays()

    image = generate_cosmic_rays(ctx, ch, cfg)

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    assert np.all(image == 0.0)
    assert len(ctx.write_image_png.calls) == 1
    _, frame_type, _ = ctx.write_image_png.calls[0]
    assert frame_type == "COSMIC_ONLY"


# With a fixed number of rays and length range, the total signal should scale with the configured charge.
def test_generate_cosmic_rays_signal_matches_config(tmp_path):
    ch = _channel()
    ctx = _ctx(tmp_path)
    cfg = _cfg_three_rays()

    image = generate_cosmic_rays(ctx, ch, cfg)

    assert image.shape == (ch.y_pixels, ch.x_pixels)

    non_zero = image[image > 0]
    assert non_zero.size > 0
    assert set(np.unique(non_zero)) == {cfg.cosmic_ray_signal_electrons}

    total_signal = float(non_zero.sum())
    expected_min = 3 * cfg.cosmic_ray_signal_electrons
    expected_max = 3 * cfg.cosmic_ray_signal_electrons * cfg.cosmic_ray_length_max_px
    assert expected_min <= total_signal <= expected_max


# Even on a 1x1 detector with long streak lengths, generation must not crash and stays within bounds.
def test_generate_cosmic_rays_small_detector_edge_case(tmp_path):
    ch = _channel(x_pixels=1, y_pixels=1)
    ctx = _ctx(tmp_path)
    cfg = _cfg_single_ray_small_detector()

    image = generate_cosmic_rays(ctx, ch, cfg)

    assert image.shape == (1, 1)
    # Single pixel should either remain zero or equal to the configured charge; both are within bounds.
    assert image[0, 0] in (0.0, cfg.cosmic_ray_signal_electrons)

