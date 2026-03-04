from pathlib import Path


def print_if_enabled(message: str, enabled: bool) -> None:
    """Print a user-facing message only when enabled."""
    if enabled:
        print(message)


def resolve_path_under(base_dir: Path, *parts: str | Path) -> Path:
    """
    Build path from base_dir + parts, resolve it, and ensure it stays under base_dir.

    Raises ValueError if the resolved path escapes base_dir (path traversal attempt).
    Returns the resolved path on success.
    """
    path = base_dir
    for p in parts:
        path = path / p
    return ensure_path_under(path, base_dir)


def ensure_path_under(path: Path, base_dir: Path) -> Path:
    """
    Ensure resolved path is under base_dir. Raises ValueError if path traversal detected.

    Use when the path is built from user/config input (e.g. sys.argv, config file).
    """
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(
            f"Path traversal: {resolved} is outside allowed base {base_resolved}"
        )
    return resolved

