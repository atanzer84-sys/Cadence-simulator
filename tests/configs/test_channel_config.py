import pytest
from pathlib import Path

from configs.channel_config import (
    ChannelConfig,
    load_channel_config,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_channel_config_ok(tmp_path):
    # Verifies that a valid config file is parsed correctly and returns a fully populated ChannelConfig.
    cfg_file = tmp_path / "channel.cfg"

    _write(
        cfg_file,
        """
        x_pixels = 2048
        y_pixels = 1024
        resolution_factor = 1.5
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = ir_effective_area.txt
        channel_name = A
        bias_offset = 0.0
        channel_name=""
        """,
    )

    cfg = load_channel_config(cfg_file)

    assert isinstance(cfg, ChannelConfig)
    assert cfg.x_pixels == 2048
    assert cfg.y_pixels == 1024
    assert cfg.resolution_factor == pytest.approx(1.5)
    assert cfg.dark_noise == pytest.approx(0.01)
    assert cfg.read_noise == pytest.approx(3.2)
    assert cfg.effective_area_file == "ir_effective_area.txt"
    assert cfg.bias_offset == 0.0
    assert cfg.channel_name == '""'

def test_missing_config_file_raises(tmp_path):
    # Verifies that loading a non-existent config file raises FileNotFoundError.
    missing = tmp_path / "missing.cfg"

    with pytest.raises(FileNotFoundError):
        load_channel_config(missing)


def test_invalid_int_value_raises_valueerror(tmp_path):
    # Verifies that a non-integer x_pixels value raises ValueError.
    cfg_file = tmp_path / "bad_int.cfg"

    _write(
        cfg_file,
        """
        x_pixels = not_an_int
        y_pixels = 1024
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = ir.txt
        """,
    )

    with pytest.raises(ValueError):
        load_channel_config(cfg_file)


def test_invalid_float_value_raises_valueerror(tmp_path):
    # Verifies that a non-float resolution_factor value raises ValueError.
    cfg_file = tmp_path / "bad_float.cfg"

    _write(
        cfg_file,
        """
        x_pixels = 2048
        y_pixels = 1024
        resolution_factor = not_a_float
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = ir.txt
        """,
    )

    with pytest.raises(ValueError):
        load_channel_config(cfg_file)


def test_comments_and_inline_comments_are_ignored(tmp_path):
    # Verifies that full-line and inline comments are ignored during key-value parsing.
    cfg_file = tmp_path / "comments.cfg"

    _write(
        cfg_file,
        """
        # detector configuration
        x_pixels = 2048      # detector width
        y_pixels = 1024
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = ir.txt   # calibration file
        channel_name = X
        """,
    )

    cfg = load_channel_config(cfg_file)

    assert cfg.x_pixels == 2048
    assert cfg.effective_area_file == "ir.txt"


def test_missing_required_key_raises_keyerror(tmp_path):
    # Verifies that if a required key is missing in the config file, a KeyError is raised.
    cfg_file = tmp_path / "missing_key.cfg"

    _write(
        cfg_file,
        """
        x_pixels = 2048
        y_pixels = 1024
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        """,
    )

    with pytest.raises(KeyError):
        load_channel_config(cfg_file)


def test_channel_config_is_frozen(tmp_path):
    # Verifies that ChannelConfig is immutable (frozen dataclass).
    cfg_file = tmp_path / "ok.cfg"

    _write(
        cfg_file,
        """
        x_pixels = 100
        y_pixels = 100
        resolution_factor = 1.0
        dark_noise = 0.01
        read_noise = 3.2
        effective_area_file = test.txt
        channel_name = A
        """,
    )

    cfg = load_channel_config(cfg_file)

    with pytest.raises(Exception):
        cfg.x_pixels = 200


def test_duplicate_keys_last_value_wins(tmp_path):
    # Verifies that when a key appears multiple times, the last occurrence overwrites earlier ones.
    cfg_file = tmp_path / "channel.cfg"
    cfg_file.write_text(
        """
        effective_area_file = ea.txt
        x_pixels = 1024
        y_pixels = 512
        resolution_factor = 1.0
        dark_noise = 0.0
        read_noise = 3.5
        channel_name = NUV
        x_pixels = 2048
        """
    )

    cfg = load_channel_config(cfg_file)

    assert cfg.x_pixels == 2048

