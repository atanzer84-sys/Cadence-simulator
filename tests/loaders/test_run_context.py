from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from loaders.run_context import RunContext


# Tests: RunContext
# Behavior: stores all provided values unchanged
def test_run_context_stores_all_fields():
    dump_3d_array = lambda *args, **kwargs: None
    dump_1d_array = lambda *args, **kwargs: None
    dump_1d_for_channel = lambda *args, **kwargs: None
    plot_1d_for_channel = lambda *args, **kwargs: None
    plot_flux_and_photons_windows = lambda *args, **kwargs: None
    plot_star_counts_vs_noise_spectroscopy = lambda *args, **kwargs: None
    plot_star_counts_vs_noise_photometry = lambda *args, **kwargs: None
    write_calibration_frame_png = lambda *args, **kwargs: None
    write_science_frame_png = lambda *args, **kwargs: None
    write_science_frame_component_png = lambda *args, **kwargs: None
    generate_background_star_visibility_on_science_frame = lambda *args, **kwargs: None

    timestamp = datetime(2025, 1, 1, 12, 0, 0)
    output_dir = Path("/tmp/test-output")

    ctx = RunContext(
        target_name="WASP-69",
        output_dir=output_dir,
        timestamp=timestamp,
        dump_3d_array=dump_3d_array,
        dump_1d_array=dump_1d_array,
        dump_1d_for_channel=dump_1d_for_channel,
        plot_1d_for_channel=plot_1d_for_channel,
        plot_flux_and_photons_windows=plot_flux_and_photons_windows,
        plot_star_counts_vs_noise_spectroscopy=plot_star_counts_vs_noise_spectroscopy,
        plot_star_counts_vs_noise_photometry=plot_star_counts_vs_noise_photometry,
        write_calibration_frame_png=write_calibration_frame_png,
        write_science_frame_png=write_science_frame_png,
        write_science_frame_component_png=write_science_frame_component_png,
        generate_background_star_visibility_on_science_frame=generate_background_star_visibility_on_science_frame,
    )

    assert ctx.target_name == "WASP-69"
    assert ctx.output_dir == output_dir
    assert ctx.timestamp == timestamp
    assert ctx.dump_3d_array is dump_3d_array
    assert ctx.dump_1d_array is dump_1d_array
    assert ctx.dump_1d_for_channel is dump_1d_for_channel
    assert ctx.plot_1d_for_channel is plot_1d_for_channel
    assert ctx.plot_flux_and_photons_windows is plot_flux_and_photons_windows
    assert ctx.plot_star_counts_vs_noise_spectroscopy is plot_star_counts_vs_noise_spectroscopy
    assert ctx.plot_star_counts_vs_noise_photometry is plot_star_counts_vs_noise_photometry
    assert ctx.write_calibration_frame_png is write_calibration_frame_png
    assert ctx.write_science_frame_png is write_science_frame_png
    assert ctx.write_science_frame_component_png is write_science_frame_component_png
    assert ctx.generate_background_star_visibility_on_science_frame is generate_background_star_visibility_on_science_frame


# Tests: RunContext
# Behavior: rejects field mutation after creation
def test_run_context_is_frozen():
    ctx = RunContext(
        target_name="WASP-69",
        output_dir=Path("/tmp/test-output"),
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        dump_3d_array=lambda *args, **kwargs: None,
        dump_1d_array=lambda *args, **kwargs: None,
        dump_1d_for_channel=lambda *args, **kwargs: None,
        plot_1d_for_channel=lambda *args, **kwargs: None,
        plot_flux_and_photons_windows=lambda *args, **kwargs: None,
        plot_star_counts_vs_noise_spectroscopy=lambda *args, **kwargs: None,
        plot_star_counts_vs_noise_photometry=lambda *args, **kwargs: None,
        write_calibration_frame_png=lambda *args, **kwargs: None,
        write_science_frame_png=lambda *args, **kwargs: None,
        write_science_frame_component_png=lambda *args, **kwargs: None,
        generate_background_star_visibility_on_science_frame=lambda *args, **kwargs: None,
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
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            dump_3d_array=lambda *args, **kwargs: None,
            dump_1d_array=lambda *args, **kwargs: None,
            dump_1d_for_channel=lambda *args, **kwargs: None,
            plot_1d_for_channel=lambda *args, **kwargs: None,
            plot_flux_and_photons_windows=lambda *args, **kwargs: None,
            plot_star_counts_vs_noise_spectroscopy=lambda *args, **kwargs: None,
            plot_star_counts_vs_noise_photometry=lambda *args, **kwargs: None,
            write_calibration_frame_png=lambda *args, **kwargs: None,
            write_science_frame_png=lambda *args, **kwargs: None,
            write_science_frame_component_png=lambda *args, **kwargs: None,
        )