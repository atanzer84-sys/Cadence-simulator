def print_if_enabled(message: str, enabled: bool) -> None:
    """Print a user-facing message only when enabled."""
    if enabled:
        print(message)
