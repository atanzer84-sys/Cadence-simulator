import pytest
from datetime import datetime

from loaders.run_context import RunContext


def _noop(*args, **kwargs):
    pass


@pytest.fixture
def make_run_context(tmp_path):
    def _make_run_context(**overrides):
        base = dict(
            target_name="HD_2685",
            output_dir=tmp_path,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            timestamp_str="20260101_120000_000000",
            dump_3d_array=_noop,
            dump_1d_array=_noop,
            dump_1d_for_channel=_noop,
            plot_1d_for_channel=_noop,
            plot_flux_and_photons_windows=_noop,
            plot_star_counts_vs_noise_spectroscopy=_noop,
            plot_star_counts_vs_noise_photometry=_noop,
            write_calibration_frame_png=_noop,
            write_science_frame_png=_noop,
            write_science_frame_component_png=_noop,
            generate_background_star_visibility_on_science_frame=_noop,
        )
        base.update(overrides)
        return RunContext(**base)

    return _make_run_context