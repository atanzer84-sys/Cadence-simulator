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

    # prevent logging setup / file creation
    monkeypatch.setattr(
        waltzer_simulator,
        "initialize_waltzer_runtime",
        lambda: tmp_path
    )

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
