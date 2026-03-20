import numpy as np
import pytest

from utils.images_calibration_frame import write_calibration_frame_png


@pytest.fixture
def calibration_png_base(make_run_context, make_global_config, make_star):
    detector_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    ctx = make_run_context(target_name="TargetA")
    cfg = make_global_config(invert_science_frames=False)
    star = make_star(name="StarA")
    return detector_data, ctx, cfg, star


@pytest.fixture
def save_call_recorder(monkeypatch):
    calls = {}

    def _make_recorder(key):
        def _save(frame_to_plot, filename, title, stats_text, channel_name=None):
            calls[key] = {
                "frame_to_plot": frame_to_plot,
                "filename": filename,
                "title": title,
                "stats_text": stats_text,
                "channel_name": channel_name,
            }
        return _save

    monkeypatch.setattr(
        "utils.images_calibration_frame.save_single_frame_png_VIS_cropped",
        _make_recorder("VIS"),
    )
    monkeypatch.setattr(
        "utils.images_calibration_frame.save_single_frame_png_NUV",
        _make_recorder("NUV"),
    )
    monkeypatch.setattr(
        "utils.images_calibration_frame.save_single_frame_png_NIR",
        _make_recorder("NIR"),
    )

    return calls


# Tests: write_calibration_frame_png
# Behavior: uses the VIS writer and forwards the original detector data
def test_write_calibration_frame_png_vis_uses_vis_writer(
    calibration_png_base,
    save_call_recorder,
    make_photometry_channel,
):
    detector_data, ctx, cfg, star = calibration_png_base
    channel = make_photometry_channel(channel_name="VIS", exposure_s=12.0)

    write_calibration_frame_png(
        detector_data=detector_data,
        frame_type="dark",
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        index=3,
    )

    assert "VIS" in save_call_recorder
    assert "NUV" not in save_call_recorder
    assert "NIR" not in save_call_recorder

    call = save_call_recorder["VIS"]
    np.testing.assert_array_equal(call["frame_to_plot"], detector_data)
    assert call["channel_name"] == "VIS"
    assert call["filename"] is not None
    assert call["title"] is not None


# Tests: write_calibration_frame_png
# Behavior: applies inversion and uses the NIR writer when inversion is enabled
def test_write_calibration_frame_png_nir_applies_inversion_and_uses_nir_writer(
    calibration_png_base,
    save_call_recorder,
    make_global_config,
    make_photometry_channel,
):
    detector_data, ctx, _, star = calibration_png_base
    cfg = make_global_config(invert_science_frames=True)
    channel = make_photometry_channel(channel_name="NIR", exposure_s=5.0)

    write_calibration_frame_png(
        detector_data=detector_data,
        frame_type="flat",
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        index=1,
    )

    assert "NIR" in save_call_recorder
    assert "VIS" not in save_call_recorder
    assert "NUV" not in save_call_recorder

    call = save_call_recorder["NIR"]
    expected = detector_data.max() - detector_data
    np.testing.assert_array_equal(call["frame_to_plot"], expected)
    assert call["channel_name"] == "NIR"
    assert call["filename"] is not None
    assert call["title"] is not None