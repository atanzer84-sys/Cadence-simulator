"""Single source of truth for test Channel-like objects. Add new params here so all tests stay green."""

from types import SimpleNamespace

# Superset of attributes used by frame, instrument, and utils tests. Override in tests as needed.
BASE_CHANNEL = {
    "channel_name": "NUV",
    "x_pixels": 10,
    "y_pixels": 8,
    "read_noise": 5.0,
    "bias_offset": 100.0,
    "dark_noise": 0.5,
    "dark_current_sigma": 2.0,
    "ccd_gain": 2.0,
    "exposure_s": 10.0,
    "spread_half_height_pix": 1,
    "mode": 1,
    "spread_profile_file": "dummy.fits",
}


def channel(**overrides):
    """Channel-like SimpleNamespace. Override any key from BASE_CHANNEL (e.g. channel_name, exposure_s)."""
    d = dict(BASE_CHANNEL)
    d.update(overrides)
    return SimpleNamespace(**d)
