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

    class DummyCfg:
        target_name = "SomeTarget"

    monkeypatch.setattr(waltzer_simulator, "load_cfg_and_user_config", lambda: DummyCfg())

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
    monkeypatch.setattr(
        waltzer_simulator,
        "calculateFluxOnEarth",
        lambda star, outdir: calls.update(flux=(star, outdir)),
    )

    waltzer_simulator.main()

    assert calls["star"] == ({"name": "Star"}, ["name"])
    assert calls["planet"] == ({"name": "Planet"}, ["name"])
    assert calls["flux"] == (star_obj, "OUTDIR")
