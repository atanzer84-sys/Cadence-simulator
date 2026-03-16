from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from types import SimpleNamespace

from loaders import load_gaia


def _dummy_cfg(async_flag: bool = False):
    """Minimal config-like object for tests (only GAIA_USE_ASYNC_JOBS is used)."""
    return SimpleNamespace(GAIA_USE_ASYNC_JOBS=async_flag)


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


def test_lookup_target_star_gaia_returns_empty_dict_when_cone_search_empty(monkeypatch):
    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: None,
    )

    out = load_gaia.lookup_target_star_gaia(
        {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
        cfg=_dummy_cfg(),
    )
    assert out == {}


def test_lookup_target_star_gaia_returns_empty_dict_when_cone_search_returns_empty_table(monkeypatch):
    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: _cone_table([]),
    )

    out = load_gaia.lookup_target_star_gaia(
        {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
        cfg=_dummy_cfg(),
    )
    assert out == {}


def test_lookup_target_star_gaia_returns_empty_dict_when_query_gaia_returns_empty(monkeypatch):
    cone = _cone_table([(1.0, 2.0, 222)])
    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id, _async: {})

    out = load_gaia.lookup_target_star_gaia(
        {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
        cfg=_dummy_cfg(),
    )
    assert out == {}


def test_lookup_target_star_gaia_returns_empty_dict_on_any_exception(monkeypatch):
    def _raise(_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False):
        raise RuntimeError("gaia down")

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", _raise)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
        missing_star=["effective_temperature"],
        cfg=_dummy_cfg(),
    )
    assert out == {}


def test_get_gaia_stellar_properties_converts_nan_to_none():
    row = {
        "Teff": float("nan"),
        "radius_sun": 1.0,
        "mass_sun": None,
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


def test_lookup_target_star_gaia_returns_only_missing_keys(monkeypatch):
    gaia_row = {
        "ra": 1.0,
        "dec": 2.0,
        "phot_g_mean_mag": 10.0,
        "Teff": 5777.0,
        "radius_sun": 1.01,
        "mass_sun": 1.0,
        "mh_gspphot": 0.1,
        "logg_gspphot": 4.4,
        "distance_gspphot": 10.0,
    }

    cone = _cone_table([(1.0, 2.0, 222)])
    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id, _async: gaia_row)

    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature", "radius"]

    out = load_gaia.lookup_target_star_gaia(star_params, missing_star=missing, cfg=_dummy_cfg())

    assert out == {
        "effective_temperature": 5777.0,
        "radius": 1.01,
    }


def test_lookup_target_star_gaia_passes_async_flag_to_helpers(monkeypatch):
    """lookup_target_star_gaia should forward cfg.GAIA_USE_ASYNC_JOBS into _gaia_cone_search and query_gaia."""
    recorded = {"cone_async": None, "query_async": None}

    cone = _cone_table([(1.0, 2.0, 222)])

    def fake_cone(center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False):
        recorded["cone_async"] = GAIA_USE_ASYNC_JOBS
        return cone

    def fake_query(source_id, GAIA_USE_ASYNC_JOBS):
        recorded["query_async"] = GAIA_USE_ASYNC_JOBS
        # minimal gaia_row so that get_gaia_stellar_properties works
        return {
            "ra": 1.0,
            "dec": 2.0,
            "phot_g_mean_mag": 10.0,
            "Teff": 5777.0,
            "radius_sun": 1.0,
            "mass_sun": None,
            "mh_gspphot": 0.0,
            "logg_gspphot": 4.5,
            "distance_gspphot": 10.0,
        }

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", fake_cone)
    monkeypatch.setattr(load_gaia, "query_gaia", fake_query)

    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature"]

    # async = True
    cfg_async = _dummy_cfg(async_flag=True)
    load_gaia.lookup_target_star_gaia(star_params, missing_star=missing, cfg=cfg_async)
    assert recorded["cone_async"] is True
    assert recorded["query_async"] is True

    # async = False
    recorded["cone_async"] = None
    recorded["query_async"] = None
    cfg_sync = _dummy_cfg(async_flag=False)
    load_gaia.lookup_target_star_gaia(star_params, missing_star=missing, cfg=cfg_sync)
    assert recorded["cone_async"] is False
    assert recorded["query_async"] is False


def test_gaia_cone_search_applies_magnitude_limit_and_async(monkeypatch):
    """_gaia_cone_search should filter by g_mag_limit and respect GAIA_USE_ASYNC_JOBS without calling real Gaia."""
    from astroquery import gaia as aq_gaia

    calls: list[str] = []

    cone = Table(
        rows=[
            (101, 1.0, 2.0, 0.1, 10.0),
            (202, 1.1, 2.1, 0.2, 21.0),
        ],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag"),
    )

    class FakeJob:
        def __init__(self, table):
            self._table = table

        def get_results(self):
            return self._table

    class FakeGaia:
        @staticmethod
        def cone_search(center, radius):
            calls.append("sync")
            return FakeJob(cone)

        @staticmethod
        def cone_search_async(center, radius):
            calls.append("async")
            return FakeJob(cone)

    # Replace astroquery.gaia.Gaia with our fake.
    monkeypatch.setattr(aq_gaia, "Gaia", FakeGaia, raising=False)

    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    # async=True: should call cone_search_async and apply magnitude filter
    result = load_gaia._gaia_cone_search(center, radius_arcsec=100.0, g_mag_limit=20.0, GAIA_USE_ASYNC_JOBS=True)
    assert calls == ["async"]
    assert result is not None
    # Only the bright star (G=10) should remain
    assert len(result) == 1
    assert int(result["source_id"][0]) == 101

def test_get_gaia_stellar_properties_does_not_drop_radius_and_mass_for_cached_csv_columns():
    row = {
        "Teff": 4656.00146484375,
        "radius_sun": 30.21190071105957,
        "mass_sun": 4.80074405670166,
        "mh_gspphot": 0.14090000092983246,
        "logg_gspphot": 2.046299934387207,
        "ra": 294.6925327559344,
        "dec": 31.24936559398053,
        "dist_pc": 2401.2685546875,
        "phot_g_mean_mag": 11.795485496520996,
        "parallax": 0.3549924624326295,
    }

    got = load_gaia.get_gaia_stellar_properties(row, log_output=False)


    assert got["radius"] is not None
    assert got["mass"] is not None

def test_get_gaia_stellar_properties_reads_cached_csv_column_names():
    row = {
        "Teff": 4656.00146484375,
        "radius_sun": 30.21190071105957,
        "mass_sun": 4.80074405670166,
        "mh_gspphot": 0.14090000092983246,
        "logg_gspphot": 2.046299934387207,
        "ra": 294.6925327559344,
        "dec": 31.24936559398053,
        "dist_pc": 2401.2685546875,
        "phot_g_mean_mag": 11.795485496520996,
        "parallax": 0.3549924624326295,
    }

    got = load_gaia.get_gaia_stellar_properties(row, log_output=False)


    assert got["effective_temperature"] == 4656.00146484375
    assert got["radius"] == 30.21190071105957
    assert got["mass"] == 4.80074405670166
    assert got["distance"] == 2401.2685546875
    assert got["gaia_magnitude"] == 11.795485496520996