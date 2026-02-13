import pytest
import waltzer_simulator

def test_excel_file_not_found_exits(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    params = input_dir / "parameters.txt"
    params.write_text(
        "target_name = HD 202772 A\n"
        "total_observation_length_h = 1\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_IR_s = 1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _: (_ for _ in ()).throw(ValueError("no excel"))
    )

    with pytest.raises(SystemExit) as exc:
        waltzer_simulator.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Input error" in out
    assert "no excel" in out

def test_main_calls_star_and_planet_constructors(monkeypatch):
    import waltzer_simulator

    # isolate main from filesystem
    monkeypatch.setattr(waltzer_simulator, "initialize_waltzer_runtime", lambda: "OUTDIR")

    class DummyUserCfg:
        target_name = "SomeTarget"

    class DummyChannelCfg:
        pass

    class DummyCal:
        pixel_scale = 1.0

    dummy_user_cfg = DummyUserCfg()
    dummy_nuv_cfg = DummyChannelCfg()
    dummy_vis_cfg = DummyChannelCfg()
    dummy_ir_cfg = DummyChannelCfg()

    # main now expects (user_cfg, nuv_cfg, vis_cfg, ir_cfg)
    monkeypatch.setattr(
        waltzer_simulator,
        "load_cfg_and_user_config",
        lambda: (dummy_user_cfg, dummy_nuv_cfg, dummy_vis_cfg, dummy_ir_cfg),
    )
    monkeypatch.setattr(waltzer_simulator, "load_instrument_calibration", lambda *a, **k: (DummyCal(), DummyCal(), DummyCal()))

    monkeypatch.setattr(
        waltzer_simulator,
        "load_stellar_and_planetary_properties",
        lambda _target: (
            {"name": "Planet"},
            {"name": "Star"},
            ["name"],
            ["name"],
        ),
    )

    calls = {"star": None, "planet": None, "flux": None}
    star_obj = object()
    planet_obj = object()

    monkeypatch.setattr(
        waltzer_simulator.Star,
        "from_params",
        staticmethod(lambda p, required_keys: calls.update(star=(p, required_keys)) or star_obj),
    )
    monkeypatch.setattr(
        waltzer_simulator.Planet,
        "from_params",
        staticmethod(lambda p, required_keys: calls.update(planet=(p, required_keys)) or planet_obj),
    )

    # one-line capture, signature-agnostic
    monkeypatch.setattr(waltzer_simulator, "calculateFluxOnEarth", lambda *a: calls.update(flux=a))

    waltzer_simulator.main()

    assert calls["star"] == ({"name": "Star"}, ["name"])
    assert calls["planet"] == ({"name": "Planet"}, ["name"])

    # stable assertions only: star object and outdir are passed somewhere
    assert calls["flux"][0] is star_obj
    assert "OUTDIR" in calls["flux"]
