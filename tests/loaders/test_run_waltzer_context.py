"""Tests for loaders.run_waltzer_context."""

import pytest
from loaders import run_waltzer_context
import logging
import sys
from pathlib import Path
from unittest.mock import patch


# Tests: get_repo_root
# Behavior: finds repo root from nested path
def test_get_repo_root_finds_repo_root(tmp_path):
    repo_root = tmp_path
    (repo_root / "src").mkdir()
    (repo_root / "input").mkdir()

    nested_file = repo_root / "a" / "b" / "c" / "file.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("", encoding="utf-8")

    got = run_waltzer_context.get_repo_root(nested_file)

    assert got == repo_root


# Tests: get_repo_root
# Behavior: raises when repo markers are missing
def test_get_repo_root_raises_when_repo_markers_missing(tmp_path):
    base_file = tmp_path / "a" / "b" / "file.py"
    base_file.parent.mkdir(parents=True)
    base_file.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_waltzer_context.get_repo_root(base_file)


# Tests: load_global_and_user_config
# Behavior: loads both config paths and returns user config
def test_load_global_and_user_config_loads_paths_and_returns_user_cfg(make_user_config, tmp_path):
    repo_root = tmp_path
    user_parameter_path = tmp_path / "input" / "params.txt"
    user_parameter_path.parent.mkdir()
    user_parameter_path.write_text("target_name = TestStar\n", encoding="utf-8")
    user_cfg = make_user_config(target_name="TestStar")

    with patch("loaders.run_waltzer_context.get_repo_root", return_value=repo_root), \
         patch("loaders.run_waltzer_context.get_user_parameter_path", return_value=user_parameter_path), \
         patch("loaders.run_waltzer_context.load_global_config") as mock_global, \
         patch("loaders.run_waltzer_context.load_user_config") as mock_user, \
         patch("loaders.run_waltzer_context.get_user_config", return_value=user_cfg):

        got = run_waltzer_context.load_global_and_user_config()

    assert got is user_cfg
    mock_global.assert_called_once_with(repo_root / "configs" / "global.cfg")
    mock_user.assert_called_once_with(user_parameter_path)


# Tests: setup_output_directory
# Behavior: raises after repeated collisions
def test_setup_output_directory_raises_after_many_collisions(tmp_path):
    from datetime import datetime
    from pathlib import Path

    fixed_time = datetime(2025, 2, 5, 12, 0, 0, 0)
    real_mkdir = Path.mkdir

    class FixedDateTime:
        @staticmethod
        def now():
            return fixed_time

    def fake_mkdir(self, parents=False, exist_ok=False):
        if exist_ok:
            return real_mkdir(self, parents=parents, exist_ok=exist_ok)
        raise FileExistsError

    with patch("loaders.run_waltzer_context.get_repo_root", return_value=tmp_path), \
         patch("loaders.run_waltzer_context.datetime", FixedDateTime), \
         patch("pathlib.Path.mkdir", fake_mkdir):

        with pytest.raises(RuntimeError):
            run_waltzer_context.setup_output_directory()


# Tests: initialize_waltzer_runtime_context
# Behavior: builds RunContext from user config and output directory
def test_initialize_waltzer_runtime_context_builds_run_context(make_user_config, tmp_path):
    from datetime import datetime

    output_dir = tmp_path / "output" / "20250101_120000_000000"
    timestamp_str = "20250101_120000_000000"
    timestamp = datetime(2025, 1, 1, 12, 0, 0)
    user_cfg = make_user_config(target_name="TestStar")

    with patch("loaders.run_waltzer_context.setup_output_directory", return_value=(output_dir, timestamp_str, timestamp)), \
         patch("loaders.run_waltzer_context.setup_logger"), \
         patch("loaders.run_waltzer_context.load_global_and_user_config", return_value=user_cfg):

        run_ctx, got_user_cfg = run_waltzer_context.initialize_waltzer_runtime_context()

    assert got_user_cfg is user_cfg
    assert run_ctx.target_name == user_cfg.target_name
    assert run_ctx.output_dir == output_dir
    assert run_ctx.timestamp == timestamp


# Tests: setup_logger
# Behavior: writes log records to file
def test_setup_logger_writes_log_records(tmp_path):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    try:
        run_waltzer_context.setup_logger(output_dir, timestamp)

        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = output_dir / f"waltzer_simulator_{timestamp}.log"

        assert log_file.exists()
        assert log_file.stat().st_size > 0
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

        root_logger.setLevel(old_level)

        for handler in old_handlers:
            root_logger.addHandler(handler)
# Tests: setup_output_directory
# Behavior: creates a timestamped output directory
def test_setup_output_directory_creates_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    output_dir, timestamp_str, _ = run_waltzer_context.setup_output_directory()

    assert output_dir.exists()
    assert output_dir.is_dir()
    assert isinstance(timestamp_str, str)
    assert len(timestamp_str) > 0
    assert timestamp_str in output_dir.name


# Tests: setup_output_directory
# Behavior: creates unique directory when timestamp collides
def test_setup_output_directory_handles_collision(monkeypatch, tmp_path):
    class FixedDateTime:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2025, 2, 5, 12, 0, 0, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "datetime", FixedDateTime)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    first_dir, ts1, _ = run_waltzer_context.setup_output_directory()
    second_dir, ts2, _ = run_waltzer_context.setup_output_directory()

    assert first_dir.exists()
    assert second_dir.exists()
    assert ts1 == ts2
    assert first_dir != second_dir


# Tests: setup_output_directory
# Behavior: prints created output directory path
def test_setup_output_directory_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)

    run_waltzer_context.setup_output_directory()
    captured = capsys.readouterr().out

    assert "Output directory created at:" in captured


# Tests: setup_logger
# Behavior: prints created log file path
def test_setup_logger_prints(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_waltzer_context.setup_logger(output_dir, timestamp)

    captured = capsys.readouterr().out
    assert "Log file created at:" in captured
    assert f"waltzer_simulator_{timestamp}.log" in captured


# Tests: setup_logger
# Behavior: creates log file in output directory
def test_setup_logger_creates_file(tmp_path):
    output_dir = tmp_path
    timestamp = "20250101_120000_000000"

    run_waltzer_context.setup_logger(output_dir, timestamp)

    log_file = output_dir / f"waltzer_simulator_{timestamp}.log"
    assert log_file.exists()


# Tests: get_user_parameter_path
# Behavior: exits with usage when too many CLI arguments are provided
def test_too_many_arguments_exits_with_usage(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "a.txt", "b.txt"])

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Usage:" in text
    assert "parameters_file" in text


# Tests: get_user_parameter_path
# Behavior: uses default input/parameters.txt when no CLI arg is given
def test_get_user_parameter_path_default_file(monkeypatch, tmp_path, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "parameters.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])

    p = run_waltzer_context.get_user_parameter_path()

    assert p.resolve() == (tmp_path / "input" / "parameters.txt").resolve()
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "parameters.txt" in (captured.out + captured.err)


# Tests: get_user_parameter_path
# Behavior: accepts custom relative parameter file path under repo root
def test_get_user_parameter_path_custom_file(monkeypatch, tmp_path, capsys):
    """Custom parameter file must be under repo root (path validation)."""
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "custom.txt").write_text("target_name = HD 202772 A", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "input/custom.txt"])

    p = run_waltzer_context.get_user_parameter_path()

    assert p.resolve() == (tmp_path / "input" / "custom.txt").resolve()
    captured = capsys.readouterr()
    assert "User parameter file loaded:" in (captured.out + captured.err)
    assert "custom.txt" in (captured.out + captured.err)


# Tests: get_user_parameter_path
# Behavior: exits when default parameter file is missing
def test_get_user_parameter_path_missing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])

    with pytest.raises(SystemExit) as exc:
        run_waltzer_context.get_user_parameter_path()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "parameter file not found" in (captured.out + captured.err).lower()


# Tests: get_user_parameter_path
# Behavior: accepts one absolute parameter file path under repo root
def test_get_user_parameter_path_one_argument_absolute_path_success(monkeypatch, tmp_path):
    """Absolute path to param file must be under repo root (path validation)."""
    (tmp_path / "input").mkdir(exist_ok=True)
    param_file = tmp_path / "input" / "params.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", str(param_file)])
    monkeypatch.chdir(tmp_path)

    got = run_waltzer_context.get_user_parameter_path()

    assert got.resolve() == param_file.resolve()


# Tests: get_user_parameter_path
# Behavior: accepts one relative parameter file path under repo root
def test_get_user_parameter_path_one_argument_relative_path_success(monkeypatch, tmp_path):
    """Relative path to param file must resolve under repo root (path validation)."""
    subdir = tmp_path / "param"
    subdir.mkdir()
    param_file = subdir / "Wasp 99.txt"
    param_file.write_text("target_name = HD 202772 A\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "param/Wasp 99.txt"])

    got = run_waltzer_context.get_user_parameter_path()

    assert got.resolve() == param_file.resolve()


# Tests: get_user_parameter_path
# Behavior: exits when no CLI arg and default file is absent
def test_get_user_parameter_path_no_argument_file_not_found_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1


# Tests: get_user_parameter_path
# Behavior: rejects traversal paths that resolve outside repo root
def test_get_user_parameter_path_rejects_path_traversal(monkeypatch, tmp_path):
    """Path outside repo root raises ValueError (path traversal protection)."""
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "../../../etc/passwd"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert "Path traversal" in str(exc_info.value) or "outside" in str(exc_info.value).lower()


# Tests: get_user_parameter_path
# Behavior: exits when one-arg path is under root but file does not exist
def test_one_argument_file_not_found_exits(monkeypatch, tmp_path, capsys):
    """Path under repo root but file missing raises SystemExit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_repo_root", lambda base_dir=None: tmp_path)
    monkeypatch.setattr("sys.argv", ["waltzer_simulator.py", "input/nonexistent_params.txt"])

    with pytest.raises(SystemExit) as exc_info:
        run_waltzer_context.get_user_parameter_path()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "not found" in text.lower()


# Tests: load_global_and_user_config
# Behavior: raises validation error for malformed user parameter values
def test_invalid_params_raises_value_error(monkeypatch, tmp_path):
    import configs.user_config as user_config

    monkeypatch.setattr(user_config, "_USER", None)

    param_file = tmp_path / "bad.txt"
    param_file.write_text(
        "target_name = Star\n"
        "total_observation_length_h = not_a_number\n"
        "exposure_NUV_s = 1\n"
        "exposure_VIS_s = 1\n"
        "exposure_NIR_s = 1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_waltzer_context, "get_user_parameter_path", lambda: Path(param_file))

    with pytest.raises(ValueError) as exc_info:
        run_waltzer_context.load_global_and_user_config()

    assert "total_observation_length_h" in str(exc_info.value)
