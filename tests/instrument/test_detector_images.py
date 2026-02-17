import numpy as np
from types import SimpleNamespace
from instrument import detector_images

def test_generate_bias_frame_zero_noise_returns_constant_frame():
    # Verifies bias frame equals bias_offset when read_noise is zero.

    cfg = SimpleNamespace(
        x_pixels=3,
        y_pixels=2,
        bias_offset=100.0,
        read_noise=0.0,
        channel_name="NUV",
    )

    frame, _ = detector_images.generate_bias_frame(cfg, header=None)

    assert frame.shape == (2, 3)
    assert np.allclose(frame, 100.0)

def test_generate_dark_frame_adds_dark_current(monkeypatch):
    # Verifies dark frame equals bias + dark_current * exptime.

    cfg = SimpleNamespace(
        x_pixels=2,
        y_pixels=2,
        bias_offset=0.0,
        read_noise=0.0,
        dark_noise=5.0,
        channel_name="VIS",
    )

    # Force bias to zero frame
    monkeypatch.setattr(
        detector_images,
        "generate_bias_frame",
        lambda c, header=None: (np.zeros((2, 2)), None),
    )

    dark, _ = detector_images.generate_dark_frame(cfg, exptime_s=10.0, header=None)

    assert np.allclose(dark, 50.0)  # 5 * 10

def test_initialize_fits_header_contains_basic_keys(monkeypatch):
    # Verifies initialize_fits_header sets telescope, RA, DEC, and target metadata.

    from types import SimpleNamespace
    from datetime import datetime

    star = SimpleNamespace(
        name="TEST",
        right_ascension=10.0,
        declination=20.0,
        distance_pc=5.0,
        mass_sun_kg=1.0,
    )

    monkeypatch.setattr(
        detector_images.loaders.run_setup,
        "GLOBAL_TIMESTAMP",
        datetime(2025, 1, 1, 12, 0, 0),
    )

    header = detector_images.initialize_fits_header(star)

    assert header["TELESCOP"] == "WALTzER"
    assert header["TARGT_ID"] == "TEST"
    assert "RA_HEX" in header
    assert "DEC_HEX" in header

def test_generate_bias_dark_frames_returns_empty_when_zero_frames(monkeypatch):
    # Verifies no frames are generated when n_bias_and_darkframes is zero.

    monkeypatch.setattr(
        detector_images,
        "get_global_config",
        lambda: SimpleNamespace(n_bias_and_darkframes=0),
    )

    nuv_cfg = SimpleNamespace()
    vis_cfg = SimpleNamespace()
    user_cfg = SimpleNamespace()

    out = detector_images.generate_bias_dark_frames(nuv_cfg, vis_cfg, user_cfg, "out", SimpleNamespace())

    assert out == ([], [], [], [])

