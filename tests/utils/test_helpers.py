"""Tests for utils.helpers."""
import logging

import pytest

from utils.helpers import (
    announce,
    ensure_path_under,
    announce,
    resolve_path_under,
)


def test_announce_enabled_true_prints(capsys):
    """When enabled=True, message is printed to stdout."""
    announce("hello", True)
    out, _ = capsys.readouterr()
    assert "hello" in out


def test_announce_enabled_false_does_not_print(capsys):
    """When enabled=False, nothing is printed."""
    announce("hello", False)
    out, _ = capsys.readouterr()
    assert "hello" not in out



def test_announce_prints_when_to_user_true(capsys):
    """announce() prints to stdout when to_user=True."""
    announce("to user", to_user=True)
    out, _ = capsys.readouterr()
    assert "to user" in out


def test_resolve_path_under_joins_parts_and_returns_resolved(tmp_path):
    """Joins segments under base, resolves, and returns the path; dir must exist."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    result = resolve_path_under(tmp_path, "a", "b")
    assert result == (tmp_path / "a" / "b").resolve()
    assert result.is_dir()


def test_resolve_path_under_traversal_raises(tmp_path):
    """Path segments that escape the base (e.g. '..') raise ValueError."""
    with pytest.raises(ValueError, match="Path traversal"):
        resolve_path_under(tmp_path, "..", "etc")


def test_ensure_path_under_under_returns_resolved(tmp_path):
    """When path is under base, returns resolved path."""
    sub = tmp_path / "sub"
    sub.mkdir()
    result = ensure_path_under(sub, tmp_path)
    assert result == sub.resolve()


def test_ensure_path_under_outside_raises(tmp_path):
    """When path is outside base, raises ValueError (path traversal)."""
    outside = tmp_path / ".." / "outside"
    with pytest.raises(ValueError, match="Path traversal"):
        ensure_path_under(outside, tmp_path)
