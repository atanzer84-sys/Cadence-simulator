import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from loaders import load_gaia


def _cone_table(rows):
    """Build a minimal cone result table with ra, dec, source_id for _find_central_row."""
    if not rows:
        return Table(names=("ra", "dec", "source_id"), dtype=(float, float, int))
    return Table(rows=rows, names=("ra", "dec", "source_id"))


def test_find_central_row_returns_nearest():
    # Three sources: (1, 2) is center, (1.001, 2) is nearest, (2, 2) farther
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([
        (2.0, 2.0, 111),
        (1.001, 2.0, 222),
        (1.0, 2.001, 333),
    ])
    row, sep_arcsec = load_gaia._find_central_row(table, center)
    assert row is not None
    assert int(row["source_id"]) == 222
    assert sep_arcsec is not None
    assert sep_arcsec < 5.0


def test_find_central_row_returns_none_for_empty_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([])
    row, sep = load_gaia._find_central_row(table, center)
    assert row is None
    assert sep is None


def test_find_central_row_returns_none_for_none_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    row, sep = load_gaia._find_central_row(None, center)
    assert row is None
    assert sep is None


def test_lookup_star_gaia_returns_empty_dict_when_cone_search_empty(monkeypatch):
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _center, radius_arcsec=2.0, g_mag_limit=None: None)

    out = load_gaia.lookup_star_gaia(
        {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
    )
    assert out == {}


def test_lookup_star_gaia_returns_empty_dict_when_cone_search_returns_empty_table(monkeypatch):
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _center, radius_arcsec=2.0, g_mag_limit=None: _cone_table([]))

    out = load_gaia.lookup_star_gaia(
        {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
    )
    assert out == {}


def test_lookup_star_gaia_returns_empty_dict_when_query_gaia_returns_empty(monkeypatch):
    cone = _cone_table([(1.0, 2.0, 222)])
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _center, radius_arcsec=2.0, g_mag_limit=None: cone)
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id: {})

    out = load_gaia.lookup_star_gaia(
        {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
    )
    assert out == {}


def test_lookup_star_gaia_returns_empty_dict_on_any_exception(monkeypatch):
    def _raise(_center, radius_arcsec=2.0, g_mag_limit=None):
        raise RuntimeError("gaia down")

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", _raise)

    out = load_gaia.lookup_star_gaia(
        {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
    )
    assert out == {}


def test_get_gaia_stellar_properties_converts_nan_to_none():
    row = {
        "teff_gspphot": float("nan"),
        "radius_gspphot": 1.0,
        "mass_flame": None,
        "mh_gspphot": 0.0,
        "logg_gspphot": 4.5,
        "ra": 1.0,
        "dec": 2.0,
        "distance_gspphot": float("nan"),
        "phot_g_mean_mag": 10.0,
    }

    out = load_gaia.get_gaia_stellar_properties(row)

    assert out["effective_temperature"] is None
    assert out["radius"] == 1.0
    assert out["mass"] is None
    assert out["distance"] is None
    assert out["gaia_magnitude"] == 10.0


def test_lookup_star_gaia_returns_only_missing_keys(monkeypatch):
    gaia_row = {
        "ra": 1.0,
        "dec": 2.0,
        "phot_g_mean_mag": 10.0,
        "teff_gspphot": 5777.0,
        "radius_gspphot": 1.01,
        "mass_flame": 1.0,
        "mh_gspphot": 0.1,
        "logg_gspphot": 4.4,
        "distance_gspphot": 10.0,
    }

    cone = _cone_table([(1.0, 2.0, 222)])
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _center, radius_arcsec=2.0, g_mag_limit=None: cone)
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id: gaia_row)

    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature", "radius"]

    out = load_gaia.lookup_star_gaia(star_params, missing_star=missing)

    assert out == {
        "effective_temperature": 5777.0,
        "radius": 1.01,
    }
