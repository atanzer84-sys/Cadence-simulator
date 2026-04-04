import numpy as np
import pytest

from utils.images_science_frame import (
    save_single_frame_png_NIR,
    save_single_frame_png_NUV,
    save_single_frame_png_VIS,
    save_single_frame_png_VIS_cropped,
    write_science_frame_png,
)


@pytest.fixture
def science_png_base(make_run_context, make_global_config, make_star):
    detector_data = np.arange(100, dtype=float).reshape(10, 10)
    ctx = make_run_context(target_name="TargetA")
    cfg = make_global_config(
        invert_science_frames=False,
        science_frame_png_crop_spectrum_region=False,
    )
    star = make_star(name="StarA")
    return detector_data, ctx, cfg, star


@pytest.fixture
def science_png_recorders(monkeypatch):
    calls = {}

    def _recorder(key):
        def _save(frame_to_plot, filename, title, stats_text, y_detector_lo=None, **kwargs):
            calls[key] = {
                "frame_to_plot": frame_to_plot,
                "filename": filename,
                "title": title,
                "stats_text": stats_text,
                "y_detector_lo": y_detector_lo,
                **kwargs,
            }
        return _save

    monkeypatch.setattr(
        "utils.images_science_frame.save_single_frame_png_VIS",
        _recorder("VIS"),
    )
    monkeypatch.setattr(
        "utils.images_science_frame.save_single_frame_png_VIS_cropped",
        _recorder("VIS_CROPPED"),
    )
    monkeypatch.setattr(
        "utils.images_science_frame.save_single_frame_png_NUV",
        _recorder("NUV"),
    )
    monkeypatch.setattr(
        "utils.images_science_frame.save_single_frame_png_NIR",
        _recorder("NIR"),
    )

    return calls


# Tests: write_science_frame_png
# Behavior: uses the VIS writer and forwards the original detector data when no crop is enabled
def test_write_science_frame_png_vis_uses_standard_vis_writer(
    science_png_base,
    science_png_recorders,
    make_photometry_channel,
):
    detector_data, ctx, cfg, star = science_png_base
    channel = make_photometry_channel(channel_name="VIS", exposure_s=12.0)

    write_science_frame_png(
        detector_data=detector_data,
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        index=2,
    )

    assert "VIS" in science_png_recorders
    assert "VIS_CROPPED" not in science_png_recorders
    assert "NUV" not in science_png_recorders
    assert "NIR" not in science_png_recorders

    call = science_png_recorders["VIS"]
    np.testing.assert_array_equal(call["frame_to_plot"], detector_data)
    assert call["y_detector_lo"] is None
    assert call["filename"] is not None
    assert call["title"] is not None
    assert call["stats_text"] is not None


# Tests: write_science_frame_png
# Behavior: applies inversion before calling the NIR writer
def test_write_science_frame_png_nir_applies_inversion(
    science_png_base,
    science_png_recorders,
    make_global_config,
    make_photometry_channel,
):
    detector_data, ctx, _, star = science_png_base
    cfg = make_global_config(
        invert_science_frames=True,
        science_frame_png_crop_spectrum_region=False,
    )
    channel = make_photometry_channel(
        channel_name="NIR",
        exposure_s=5.0,
        draw_aperture_photometry_overlay=False,
    )

    write_science_frame_png(
        detector_data=detector_data,
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        phot=None,
        index=1,
    )

    assert "NIR" in science_png_recorders
    call = science_png_recorders["NIR"]

    expected = detector_data.max() - detector_data
    np.testing.assert_array_equal(call["frame_to_plot"], expected)
    assert call["phot"] is None
    assert call["draw_aperture_photometry_overlay"] is False


# Tests: write_science_frame_png
# Behavior: crops the spectroscopy strip and uses the cropped VIS writer when crop is enabled
def test_write_science_frame_png_vis_spectroscopy_uses_cropped_writer(
    science_png_base,
    science_png_recorders,
    make_global_config,
    make_spectroscopy_channel,
    monkeypatch,
):
    detector_data, ctx, _, star = science_png_base
    cfg = make_global_config(
        invert_science_frames=False,
        science_frame_png_crop_spectrum_region=True,
    )
    channel = make_spectroscopy_channel(
        channel_name="VIS",
        exposure_s=20.0,
        y_pixels=10,
        slit_half_length_arcsec=2.0,
        pixel_scale=1.0,
    )

    monkeypatch.setattr(
        "utils.images_science_frame.get_target_star_detector_position",
        lambda channel: (5, 4),
    )

    write_science_frame_png(
        detector_data=detector_data,
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        index=7,
    )

    assert "VIS_CROPPED" in science_png_recorders
    assert "VIS" not in science_png_recorders

    call = science_png_recorders["VIS_CROPPED"]

    # slit_half_length_arcsec / pixel_scale = 2
    # strip_height = 2 * 2 + 1 = 5
    # margin = round(0.2 * 5) = 1
    # half_height = 2 + 1 = 3
    # y0 = 4  -> rows 1:8
    expected = detector_data[1:8, :]
    np.testing.assert_array_equal(call["frame_to_plot"], expected)
    assert call["y_detector_lo"] == 1

# Tests: write_science_frame_png
# Behavior: applies inversion and forwards photometry inputs to the NIR writer
def test_write_science_frame_png_nir_applies_inversion_and_forwards_photometry(
    science_png_base,
    science_png_recorders,
    make_global_config,
    make_photometry_channel,
):
    detector_data, ctx, _, star = science_png_base
    cfg = make_global_config(
        invert_science_frames=True,
        science_frame_png_crop_spectrum_region=False,
    )
    channel = make_photometry_channel(
        channel_name="NIR",
        exposure_s=5.0,
        draw_aperture_photometry_overlay=True,
    )

    phot = object()

    write_science_frame_png(
        detector_data=detector_data,
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        phot=phot,
        index=1,
    )

    assert "NIR" in science_png_recorders

    call = science_png_recorders["NIR"]
    expected = detector_data.max() - detector_data
    np.testing.assert_array_equal(call["frame_to_plot"], expected)
    assert call["phot"] is phot
    assert call["draw_aperture_photometry_overlay"] is True

# Tests: write_science_frame_png
# Behavior: uses the NUV writer for NUV channels.
def test_write_science_frame_png_nuv_uses_nuv_writer(
    science_png_base,
    science_png_recorders,
    make_spectroscopy_channel,
):
    detector_data, ctx, cfg, star = science_png_base
    channel = make_spectroscopy_channel(channel_name="NUV", exposure_s=15.0, x_pixels=10, y_pixels=10)

    write_science_frame_png(
        detector_data=detector_data,
        channel=channel,
        ctx=ctx,
        cfg=cfg,
        star=star,
        index=4,
    )

    assert "NUV" in science_png_recorders
    assert "VIS" not in science_png_recorders
    assert "VIS_CROPPED" not in science_png_recorders
    assert "NIR" not in science_png_recorders

    call = science_png_recorders["NUV"]
    np.testing.assert_array_equal(call["frame_to_plot"], detector_data)
    assert call["filename"] is not None
    assert call["title"] is not None
    assert call["stats_text"] is not None


# Tests: save_single_frame_png_NIR
# Behavior: writes PNG and supports aperture overlay with photometry tuple.
def test_save_single_frame_png_nir_writes_png_with_aperture_overlay(tmp_path):
    array = np.arange(100, dtype=float).reshape(10, 10)
    filename = tmp_path / "nir_science.png"
    title = "NIR Science"
    stats_text = "MEDIAN=10.0\nSTD=2.0"
    phot = (1200.0, 35.0, 5.0, 5.0, 2.0, 3.0)

    save_single_frame_png_NIR(
        array=array,
        filename=filename,
        title=title,
        stats_text=stats_text,
        phot=phot,
        draw_aperture_photometry_overlay=True,
    )

    assert filename.exists()
    assert filename.stat().st_size > 0


# Tests: save_single_frame_png_NUV
# Behavior: writes a PNG with title and stats text for NUV frames.
def test_save_single_frame_png_nuv_writes_png(tmp_path):
    array = np.arange(64, dtype=float).reshape(8, 8)
    filename = tmp_path / "nuv_science.png"
    title = "NUV Science"
    stats_text = "MEAN=8.0\nSTD=1.2"

    save_single_frame_png_NUV(
        array=array,
        filename=filename,
        title=title,
        stats_text=stats_text,
    )

    assert filename.exists()
    assert filename.stat().st_size > 0


# Tests: save_single_frame_png_VIS
# Behavior: writes a PNG with title and stats text for VIS frames.
def test_save_single_frame_png_vis_writes_png(tmp_path):
    array = np.arange(81, dtype=float).reshape(9, 9)
    filename = tmp_path / "vis_science.png"
    title = "VIS Science"
    stats_text = "MEAN=9.0\nSTD=1.5"

    save_single_frame_png_VIS(
        array=array,
        filename=filename,
        title=title,
        stats_text=stats_text,
    )

    assert filename.exists()
    assert filename.stat().st_size > 0


# Tests: save_single_frame_png_VIS_cropped
# Behavior: writes a PNG for cropped VIS frame rendering.
def test_save_single_frame_png_vis_cropped_writes_png(tmp_path):
    array = np.arange(30, dtype=float).reshape(5, 6)
    filename = tmp_path / "vis_science_cropped.png"
    title = "VIS Cropped"
    stats_text = "MED=4.0\nSTD=0.9"

    save_single_frame_png_VIS_cropped(
        array=array,
        filename=filename,
        title=title,
        stats_text=stats_text,
    )

    assert filename.exists()
    assert filename.stat().st_size > 0