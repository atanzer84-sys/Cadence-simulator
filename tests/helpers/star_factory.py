"""Single source of truth for test Star-like objects. Add or rename params here so all tests stay green."""

from types import SimpleNamespace

# Attributes used by images_common, target_background_star_vs_noise, flux_image_array tests.
# If the domain Star renames an attribute (e.g. distance_pc), change it here once.
BASE_STAR = {
    "name": "HD 2685",
    "effective_temperature": 5778.0,
    "distance_pc": 100.0,
    "gaia_magnitude": None,
    "mass": 1.1,
}


def star(**overrides):
    """Star-like SimpleNamespace. Override any key from BASE_STAR (e.g. name, distance_pc, gaia_magnitude)."""
    d = dict(BASE_STAR)
    d.update(overrides)
    return SimpleNamespace(**d)
