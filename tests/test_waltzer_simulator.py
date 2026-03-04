from datetime import datetime
from types import SimpleNamespace

import pytest

import waltzer_simulator
from loaders.run_waltzer_context import _NOOP, _NOOP_PLOTS


def test_main_catches_exception_exits_with_message(monkeypatch, tmp_path, capsys):
    """main() catches any exception, prints 'Input error: {e}', and exits with code 1."""
    run_ctx = SimpleNamespace(
        target_name="HD 202772 A",
        output_dir=tmp_path,
        timestamp=datetime.now(),
        timestamp_str="20250101_120000",
        test_mode=_NOOP,
        produce_plots=_NOOP_PLOTS,
    )
    user_cfg = SimpleNamespace(
        target_name="HD 202772 A",
        exposure_NUV_s=1.0,
        exposure_VIS_s=1.0,
        exposure_IR_s=1.0,
    )

    def fake_init():
        return run_ctx, user_cfg

    def fake_load_channels(_user_cfg, _run_ctx):
        return SimpleNamespace(), SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(waltzer_simulator, "initialize_waltzer_runtime_context", fake_init)
    monkeypatch.setattr(waltzer_simulator, "load_channels_config", fake_load_channels)
    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _: (_ for _ in ()).throw(ValueError("no excel")),
    )

    with pytest.raises(SystemExit) as exc:
        waltzer_simulator.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Input error" in out
    assert "no excel" in out
