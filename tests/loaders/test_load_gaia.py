from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import pytest

from loaders import load_gaia


def _cone_table(rows):
    if not rows:
        return Table(names=("ra", "dec", "source_id"), dtype=(float, float, int))
    return Table(rows=rows, names=("ra", "dec", "source_id"))


def _make_gaia_row(**overrides):
    row = {
        "ra": 286.97138676797084,
        "dec": 46.86824405830706,
        "parallax": 2.031923106187413,
        "phot_g_mean_mag": 10.38234,
        "Teff": 5777.0,
        "dist_pc": 10.0,
        "radius_sun": 1.0,
        "mass_sun": 1.0,
        "mh_gspphot": 0.0,
        "logg_gspphot": 4.5,
    }
    row.update(overrides)
    return row


# Tests: _find_central_row
# Behavior: returns the nearest source to the center position
def test_find_central_row_returns_nearest():
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


# Tests: _find_central_row
# Behavior: returns None values for an empty table
def test_find_central_row_returns_none_for_empty_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([])

    row, sep = load_gaia._find_central_row(table, center)

    assert row is None
    assert sep is None


# Tests: _find_central_row
# Behavior: returns None values for a missing table
def test_find_central_row_returns_none_for_none_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    row, sep = load_gaia._find_central_row(None, center)

    assert row is None
    assert sep is None


# Tests: lookup_target_star_gaia
# Behavior: raises when cone search returns None
def test_lookup_target_star_gaia_raises_when_cone_search_empty(monkeypatch, make_global_config):
    import pytest

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: None,
    )

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="No Gaia cone result found"):
        load_gaia.lookup_target_star_gaia(
            {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: raises when cone search returns no rows
def test_lookup_target_star_gaia_raises_when_cone_search_returns_empty_table(monkeypatch, make_global_config):
    import pytest

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: _cone_table([]),
    )

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="No Gaia cone result found"):
        load_gaia.lookup_target_star_gaia(
            {"name": "No Match", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: raises when Gaia row lookup returns no data
def test_lookup_target_star_gaia_raises_when_query_gaia_returns_empty(monkeypatch, make_global_config):
    import pytest

    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id, _async: {})

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="No Gaia row returned"):
        load_gaia.lookup_target_star_gaia(
            {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: propagates errors when cone search raises
def test_lookup_target_star_gaia_raises_on_any_exception(monkeypatch, make_global_config):
    import pytest

    def _raise(_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False):
        raise RuntimeError("gaia down")

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", _raise)

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="gaia down"):
        load_gaia.lookup_target_star_gaia(
            {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: returns only requested missing keys from Gaia data
def test_lookup_target_star_gaia_returns_only_missing_keys(monkeypatch, make_global_config):
    gaia_row = _make_gaia_row(Teff=5777.0, radius_sun=1.01, mass_sun=1.0, mh_gspphot=0.1)

    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda _source_id, _async: gaia_row)

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature", "radius"]

    out = load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg)

    assert out == {
        "effective_temperature": 5777.0,
        "radius": 1.01,
    }


# Tests: lookup_target_star_gaia
# Behavior: forwards the async flag to Gaia helpers
def test_lookup_target_star_gaia_passes_async_flag_to_helpers(monkeypatch, make_global_config):
    recorded = {"cone_async": None, "query_async": None}
    cone = _cone_table([(1.0, 2.0, 222)])

    def fake_cone(center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False):
        recorded["cone_async"] = GAIA_USE_ASYNC_JOBS
        return cone

    def fake_query(source_id, GAIA_USE_ASYNC_JOBS):
        recorded["query_async"] = GAIA_USE_ASYNC_JOBS
        return _make_gaia_row()

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", fake_cone)
    monkeypatch.setattr(load_gaia, "query_gaia", fake_query)

    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature"]

    cfg_async = make_global_config(GAIA_USE_ASYNC_JOBS=True)
    load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg_async)
    assert recorded["cone_async"] is True
    assert recorded["query_async"] is True

    recorded["cone_async"] = None
    recorded["query_async"] = None

    cfg_sync = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg_sync)
    assert recorded["cone_async"] is False
    assert recorded["query_async"] is False


# Tests: get_gaia_stellar_properties
# Behavior: converts NaN values in Gaia fields to None
def test_get_gaia_stellar_properties_converts_nan_to_none():
    row = _make_gaia_row(Teff=float("nan"), dist_pc=float("nan"), mass_sun=None)

    out = load_gaia.get_gaia_stellar_properties(row)

    assert out["effective_temperature"] is None
    assert out["radius"] == 1.0
    assert out["mass"] is None
    assert out["distance"] is None
    assert out["gaia_magnitude"] == 10.38234


# Tests: get_gaia_stellar_properties
# Behavior: reads Gaia cached CSV column names into internal keys
def test_get_gaia_stellar_properties_reads_cached_csv_column_names():
    row = _make_gaia_row(
        Teff=4656.00146484375,
        radius_sun=30.21190071105957,
        mass_sun=4.80074405670166,
        mh_gspphot=0.14090000092983246,
        logg_gspphot=2.046299934387207,
        ra=294.6925327559344,
        dec=31.24936559398053,
        dist_pc=2401.2685546875,
        phot_g_mean_mag=11.795485496520996,
        parallax=0.3549924624326295,
    )

    got = load_gaia.get_gaia_stellar_properties(row, log_output=False)

    assert got["effective_temperature"] == 4656.00146484375
    assert got["radius"] == 30.21190071105957
    assert got["mass"] == 4.80074405670166
    assert got["distance"] == 2401.2685546875
    assert got["gaia_magnitude"] == 11.795485496520996


# Tests: get_gaia_stellar_properties
# Behavior: converts empty strings in numeric Gaia fields to None
def test_get_gaia_stellar_properties_handles_empty_strings_from_gaia():
    gaia_row = _make_gaia_row(
        Teff="",
        dist_pc="",
        radius_sun="",
        mass_sun="",
        mh_gspphot=np.ma.masked,
        logg_gspphot=np.ma.masked,
    )

    result = load_gaia.get_gaia_stellar_properties(gaia_row, log_output=False)

    assert result["parallax"] == 2.031923106187413
    assert result["gaia_magnitude"] == 10.38234
    assert result["effective_temperature"] is None
    assert result["distance"] is None
    assert result["radius"] is None
    assert result["mass"] is None
    assert result["metallicity"] is None
    assert result["surface_gravity"] is None


# Tests: get_gaia_stellar_properties
# Behavior: converts whitespace-only numeric Gaia fields to None
def test_get_gaia_stellar_properties_handles_whitespace_strings():
    gaia_row = _make_gaia_row(
        Teff="   ",
        dist_pc="   ",
        radius_sun="   ",
        mass_sun="   ",
        mh_gspphot="   ",
        logg_gspphot="   ",
    )

    result = load_gaia.get_gaia_stellar_properties(gaia_row, log_output=False)

    assert result["parallax"] == 2.031923106187413
    assert result["gaia_magnitude"] == 10.38234
    assert result["effective_temperature"] is None
    assert result["distance"] is None
    assert result["radius"] is None
    assert result["mass"] is None
    assert result["metallicity"] is None
    assert result["surface_gravity"] is None

# Tests: lookup_target_star_gaia
# Behavior: resolves target coordinates by name when RA and Dec are missing
def test_lookup_target_star_gaia_resolves_name_when_coordinates_missing(monkeypatch, make_global_config):
    center = SkyCoord(ra=11.0 * u.deg, dec=22.0 * u.deg, frame="icrs")
    cone = _cone_table([(11.0, 22.0, 222)])

    monkeypatch.setattr(load_gaia.SkyCoord, "from_name", staticmethod(lambda name: center))
    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda found_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone if found_center is center else None,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda source_id, async_flag: _make_gaia_row(Teff=5000.0))

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "Target Without Coordinates"},
        missing_stellar_keys=["effective_temperature"],
        cfg=cfg,
    )

    assert out == {"effective_temperature": 5000.0}


# Tests: lookup_target_star_gaia
# Behavior: raises when central row cannot be identified
def test_lookup_target_star_gaia_raises_when_find_central_row_returns_none(monkeypatch, make_global_config):
    import pytest

    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda found_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "_find_central_row", lambda cone_table, found_center: (None, None))

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="No Gaia central match found"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Target", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: raises when requested keys are absent from Gaia properties
def test_lookup_target_star_gaia_raises_when_requested_keys_are_absent(monkeypatch, make_global_config):
    import pytest

    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda found_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda source_id, async_flag: _make_gaia_row())
    monkeypatch.setattr(load_gaia, "get_gaia_stellar_properties", lambda gaia_row: {"mass": 1.0})

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="did not return requested missing keys"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Target", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature", "radius"],
            cfg=cfg,
        )

# Tests: lookup_target_star_gaia
# Behavior: returns an empty dict when no missing keys are requested
def test_lookup_target_star_gaia_returns_empty_dict_when_missing_star_is_empty(monkeypatch, make_global_config):
    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda found_center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: cone,
    )
    monkeypatch.setattr(load_gaia, "query_gaia", lambda source_id, async_flag: _make_gaia_row())

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "Target", "right_ascension": 1.0, "declination": 2.0},
        missing_stellar_keys=[],
        cfg=cfg,
    )

    assert out == {}


# Tests: query_gaia
# Behavior: returns the first Gaia result row when the TAP job has data
def test_query_gaia_returns_first_row(monkeypatch):
    result = Table(
        rows=[(123, 1.0, 2.0)],
        names=("source_id", "ra", "dec"),
    )
    recorded = {"query": None, "async": None}

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")

    def fake_run(query, async_flag):
        recorded["query"] = query
        recorded["async"] = async_flag
        return result

    monkeypatch.setattr(load_gaia, "_run_gaia_job", fake_run)

    out = load_gaia.query_gaia(123, False)

    assert int(out["source_id"]) == 123
    assert float(out["ra"]) == 1.0
    assert recorded == {"query": "SELECT 123", "async": False}


# Tests: query_gaia
# Behavior: returns an empty dict when the TAP job has no rows
def test_query_gaia_returns_empty_dict_for_empty_result(monkeypatch):
    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")
    monkeypatch.setattr(load_gaia, "_run_gaia_job", lambda query, async_flag: Table(names=("source_id",), dtype=(int,)))

    out = load_gaia.query_gaia(123, False)

    assert out == {}


# Tests: query_gaia
# Behavior: propagates TAP job failures
def test_query_gaia_propagates_run_gaia_job_failure(monkeypatch):
    import pytest

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")

    def _raise(query, async_flag):
        raise RuntimeError("boom")

    monkeypatch.setattr(load_gaia, "_run_gaia_job", _raise)

    with pytest.raises(RuntimeError):
        load_gaia.query_gaia(123, False)


# Tests: _to_float
# Behavior: returns the numeric value unchanged for plain numbers
def test_to_float_returns_plain_numeric_value():
    assert load_gaia._to_float(12.5) == 12.5


# Tests: _to_float
# Behavior: returns None for blank strings
def test_to_float_returns_none_for_blank_string():
    assert load_gaia._to_float("   ") is None


# Tests: _to_float
# Behavior: returns None for NaN values
def test_to_float_returns_none_for_nan():
    assert load_gaia._to_float(float("nan")) is None


# Tests: _to_float
# Behavior: returns None for non numeric strings
def test_to_float_returns_none_for_non_numeric_string():
    assert load_gaia._to_float("not_a_number") is None


# Tests: _gaia_cone_search
# Behavior: returns sliced Gaia columns without magnitude filtering when no limit is set
def test_gaia_cone_search_returns_small_table_without_magnitude_limit(monkeypatch):
    from astroquery import gaia as aq_gaia

    cone = Table(
        rows=[
            (101, 1.0, 2.0, 0.1, 10.0, 99.0),
            (202, 1.1, 2.1, 0.2, 21.0, 88.0),
        ],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "extra_column"),
    )

    class FakeJob:
        def __init__(self, table):
            self._table = table

        def get_results(self):
            return self._table

    class FakeGaia:
        @staticmethod
        def cone_search(center, radius):
            return FakeJob(cone)

        @staticmethod
        def cone_search_async(center, radius):
            raise AssertionError("async path not expected")

    monkeypatch.setattr(aq_gaia, "Gaia", FakeGaia, raising=False)

    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    result = load_gaia._gaia_cone_search(
        center,
        radius_arcsec=100.0,
        g_mag_limit=None,
        GAIA_USE_ASYNC_JOBS=False,
    )

    assert result is not None
    assert result.colnames == ["source_id", "ra", "dec", "parallax", "phot_g_mean_mag"]
    assert len(result) == 2
    assert list(result["source_id"]) == [101, 202]


# Tests: _gaia_drop_central_star
# Behavior: removes the nearest source and keeps the background stars
def test_gaia_drop_central_star_removes_central_source():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    cone = Table(
        rows=[
            (111, 1.0, 2.0, 0.1, 10.0),
            (222, 1.01, 2.0, 0.2, 11.0),
            (333, 1.02, 2.0, 0.3, 12.0),
        ],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag"),
    )

    out = load_gaia._gaia_drop_central_star(cone, center)

    assert out is not None
    field_cone, central_row, central_sep = out
    assert int(central_row["source_id"]) == 111
    assert central_sep == 0.0
    assert list(field_cone["source_id"]) == [222, 333]


# Tests: gaia_lookup_for_background_stars
# Behavior: returns None when the cone search has no sources
def test_gaia_lookup_for_background_stars_returns_none_for_empty_cone(monkeypatch, make_star):
    star = make_star(right_ascension=1.0, declination=2.0)

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: None,
    )

    out = load_gaia.gaia_lookup_for_background_stars(
        star,
        g_mag_limit=15.0,
        GAIA_USE_ASYNC_JOBS=False,
        radius_arcsec=30.0,
    )

    assert out is None


# Tests: gaia_lookup_for_background_stars
# Behavior: returns None when target coordinates are missing
def test_gaia_lookup_for_background_stars_returns_none_when_target_coordinates_missing(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=None, declination=None)
    called = {"cone": 0}

    def _fake_cone(*args, **kwargs):
        called["cone"] += 1
        return None

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", _fake_cone)

    out = load_gaia.gaia_lookup_for_background_stars(
        star,
        g_mag_limit=15.0,
        GAIA_USE_ASYNC_JOBS=False,
        radius_arcsec=30.0,
    )

    assert out is None
    assert called["cone"] == 0


# Tests: gaia_lookup_for_background_stars
# Behavior: removes the central star and returns joined background rows
def test_gaia_lookup_for_background_stars_happy_path_returns_joined_field_rows(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=1.0, declination=2.0)
    cone = Table(
        rows=[
            (111, 1.0, 2.0, 0.1, 10.0),
            (222, 1.01, 2.0, 0.2, 11.0),
        ],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag"),
    )
    joined = Table(
        rows=[(222, 1.01, 2.0, 0.2, 11.0, 5000.0)],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag", "Teff"),
    )

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone,
    )
    monkeypatch.setattr(
        load_gaia,
        "_gaia_fetch_ap_and_join",
        lambda field_cone, GAIA_USE_ASYNC_JOBS: joined if list(field_cone["source_id"]) == [222] else None,
    )

    out = load_gaia.gaia_lookup_for_background_stars(
        star,
        g_mag_limit=15.0,
        GAIA_USE_ASYNC_JOBS=False,
        radius_arcsec=30.0,
    )

    assert out is not None
    assert len(out) == 1
    assert list(out["source_id"]) == [222]
    assert float(out["Teff"][0]) == 5000.0


# Tests: gaia_lookup_for_background_stars
# Behavior: passes the magnitude limit into the cone search
def test_gaia_lookup_for_background_stars_forwards_magnitude_limit(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=1.0, declination=2.0)
    recorded = {"g_mag_limit": None}
    cone = Table(
        rows=[
            (111, 1.0, 2.0, 0.1, 10.0),
            (222, 1.01, 2.0, 0.2, 11.0),
        ],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag"),
    )
    joined = Table(
        rows=[(222, 1.01, 2.0, 0.2, 11.0)],
        names=("source_id", "ra", "dec", "parallax", "phot_g_mean_mag"),
    )

    def fake_cone(center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS):
        recorded["g_mag_limit"] = g_mag_limit
        return cone

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", fake_cone)
    monkeypatch.setattr(
        load_gaia,
        "_gaia_fetch_ap_and_join",
        lambda field_cone, GAIA_USE_ASYNC_JOBS: joined,
    )

    out = load_gaia.gaia_lookup_for_background_stars(
        star,
        g_mag_limit=15.5,
        GAIA_USE_ASYNC_JOBS=False,
        radius_arcsec=30.0,
    )

    assert out is not None
    assert recorded["g_mag_limit"] == 15.5


# Tests: _run_gaia_job
# Behavior: dispatches to sync/async Gaia launch based on flag
@pytest.mark.parametrize("use_async,expected_path", [(False, "sync"), (True, "async")])
def test_run_gaia_job_dispatches_sync_and_async(monkeypatch, use_async, expected_path):
    from astroquery import gaia as aq_gaia

    class FakeJob:
        def __init__(self, tag):
            self._tag = tag

        def get_results(self):
            return Table(rows=[(self._tag,)], names=("mode",))

    class FakeGaia:
        @staticmethod
        def launch_job(query):
            return FakeJob("sync")

        @staticmethod
        def launch_job_async(query):
            return FakeJob("async")

    monkeypatch.setattr(aq_gaia, "Gaia", FakeGaia, raising=False)

    out = load_gaia._run_gaia_job("SELECT 1", use_async)
    assert len(out) == 1
    assert out["mode"][0] == expected_path


# Tests: _gaia_fetch_ap_and_join
# Behavior: queries AP in batches and left-joins all returned rows
def test_gaia_fetch_ap_and_join_batches_and_joins(monkeypatch):
    field_cone = Table(
        rows=[(101, 1.0), (202, 2.0), (303, 3.0)],
        names=("source_id", "ra"),
    )
    recorded_chunks = []

    def fake_query_for_ids(ids):
        recorded_chunks.append(list(ids))
        return f"Q:{','.join(map(str, ids))}"

    def fake_run(query, async_flag):
        ids = [int(x) for x in query.split(":")[1].split(",")]
        return Table(rows=[(i, float(i) + 0.5) for i in ids], names=("source_id", "Teff"))

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_ids", fake_query_for_ids)
    monkeypatch.setattr(load_gaia, "_run_gaia_job", fake_run)

    out = load_gaia._gaia_fetch_ap_and_join(field_cone, GAIA_USE_ASYNC_JOBS=False, ap_batch_size=2)

    assert recorded_chunks == [[101, 202], [303]]
    assert out is not None
    assert len(out) == 3
    assert "Teff" in out.colnames


# Tests: _gaia_fetch_ap_and_join
# Behavior: returns None when one AP chunk query fails
def test_gaia_fetch_ap_and_join_returns_none_on_chunk_failure(monkeypatch):
    field_cone = Table(
        rows=[(101, 1.0), (202, 2.0), (303, 3.0)],
        names=("source_id", "ra"),
    )

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_ids", lambda ids: f"Q:{','.join(map(str, ids))}")

    def fake_run(query, async_flag):
        if "303" in query:
            raise RuntimeError("chunk failed")
        ids = [int(x) for x in query.split(":")[1].split(",")]
        return Table(rows=[(i, float(i) + 0.5) for i in ids], names=("source_id", "Teff"))

    monkeypatch.setattr(load_gaia, "_run_gaia_job", fake_run)

    out = load_gaia._gaia_fetch_ap_and_join(field_cone, GAIA_USE_ASYNC_JOBS=False, ap_batch_size=2)
    assert out is None


# Tests: _gaia_cone_search
# Behavior: returns None when magnitude filter removes all rows
def test_gaia_cone_search_returns_none_when_mag_filter_empties_result(monkeypatch):
    from astroquery import gaia as aq_gaia

    cone = Table(
        rows=[
            (101, 1.0, 2.0, 0.1, 20.0),
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
            return FakeJob(cone)

        @staticmethod
        def cone_search_async(center, radius):
            raise AssertionError("async path not expected")

    monkeypatch.setattr(aq_gaia, "Gaia", FakeGaia, raising=False)

    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    out = load_gaia._gaia_cone_search(center, radius_arcsec=100.0, g_mag_limit=10.0, GAIA_USE_ASYNC_JOBS=False)
    assert out is None