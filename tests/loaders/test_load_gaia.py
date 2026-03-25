from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import pytest

from loaders import load_gaia
from loaders.load_stellar_and_planetary_properties import merge_gaia_into_star_params


def _cone_table(rows):
    if not rows:
        return Table(names=("ra", "dec", "source_id"), dtype=(float, float, int))
    return Table(rows=rows, names=("ra", "dec", "source_id"))


def _make_gaia_row(**overrides):
    row = {
        "source_id": 222,
        "right_ascension": 286.97138676797084,
        "declination": 46.86824405830706,
        "parallax": 2.031923106187413,
        "gaia_magnitude": 10.38234,
        "effective_temperature": 5777.0,
        "distance": 10.0,
        "radius": 1.0,
        "mass": 1.0,
        "metallicity": 0.0,
        "surface_gravity": 4.5,
    }
    row.update(overrides)
    return row


class _FakeSimbadRow:
    def __init__(self, ids_value):
        self._ids_value = ids_value

    def __getitem__(self, key):
        if key == "ids":
            return [self._ids_value]
        raise KeyError(key)


class _FakeSimbadResult:
    def __init__(self, ids_value, n_rows=1):
        self._row = _FakeSimbadRow(ids_value)
        self._n_rows = n_rows

    def __len__(self):
        return self._n_rows

    def __getitem__(self, idx):
        if idx == "ids":
            return [self._row["ids"][0]]
        if idx != 0:
            raise IndexError(idx)
        return self._row



# Tests: _get_source_id
# Behavior: returns Gaia source_id from SIMBAD name resolution when target name is WASP-14
def test_get_source_id_returns_gaia_id_from_name_for_wasp_14(monkeypatch):
    expected_source_id = 1242084170974175232

    monkeypatch.setattr(
        load_gaia,
        "_resolve_gaia_source_id_from_name",
        lambda target_name: expected_source_id if target_name == "WASP-14" else None,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("_resolve_source_id_from_coordinates should not be called when name resolution succeeds")

    monkeypatch.setattr(load_gaia, "_resolve_source_id_from_coordinates", fail_if_called)

    out = load_gaia._get_source_id(
        target_name="WASP-14",
        right_ascension=None,
        declination=None,
        GAIA_USE_ASYNC_JOBS=False,
    )

    assert out == expected_source_id


# Tests: _get_source_id
# Behavior: falls back to RA/Dec coordinate lookup when name resolution fails and returns the Gaia source_id for WASP-14
def test_get_source_id_falls_back_to_coordinates_when_name_resolution_fails(monkeypatch):
    expected_source_id = 1242084170974175232
    expected_ra = 218.27662916581795
    expected_dec = 21.89468654650299

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _target_name: None)

    recorded = {}

    def fake_resolve_source_id_from_coordinates(target_name, right_ascension, declination, GAIA_USE_ASYNC_JOBS):
        recorded["target_name"] = target_name
        recorded["right_ascension"] = right_ascension
        recorded["declination"] = declination
        recorded["GAIA_USE_ASYNC_JOBS"] = GAIA_USE_ASYNC_JOBS
        return expected_source_id

    monkeypatch.setattr(load_gaia, "_resolve_source_id_from_coordinates", fake_resolve_source_id_from_coordinates)

    out = load_gaia._get_source_id(
        target_name="1242084170974175232",
        right_ascension=expected_ra,
        declination=expected_dec,
        GAIA_USE_ASYNC_JOBS=False,
    )

    assert out == expected_source_id
    assert recorded == {
        "target_name": "1242084170974175232",
        "right_ascension": expected_ra,
        "declination": expected_dec,
        "GAIA_USE_ASYNC_JOBS": False,
    }





# Tests: _resolve_gaia_source_id_from_name
# Behavior: returns Gaia source_id when SIMBAD IDS contains Gaia DR3 id
def test_resolve_gaia_source_id_from_name_success(monkeypatch):
    class FakeSimbad:
        def add_votable_fields(self, *_args, **_kwargs):
            return None

        def query_object(self, _target_name):
            return _FakeSimbadResult("Gaia DR3 1827242816201846144|HD 189733")

    import astroquery.simbad as aq_simbad
    monkeypatch.setattr(aq_simbad, "Simbad", FakeSimbad, raising=False)

    out = load_gaia._resolve_gaia_source_id_from_name("HD 189733")
    assert out == 1827242816201846144


# Tests: _resolve_gaia_source_id_from_name
# Behavior: returns None when SIMBAD has no Gaia id entry
def test_resolve_gaia_source_id_from_name_no_match_returns_none(monkeypatch):
    class FakeSimbad:
        def add_votable_fields(self, *_args, **_kwargs):
            return None

        def query_object(self, _target_name):
            return _FakeSimbadResult("HD 189733|HIP 98505")

    import astroquery.simbad as aq_simbad
    monkeypatch.setattr(aq_simbad, "Simbad", FakeSimbad, raising=False)

    out = load_gaia._resolve_gaia_source_id_from_name("HD 189733")
    assert out is None


# Tests: _resolve_gaia_source_id_from_name
# Behavior: returns None when SIMBAD query raises an exception
def test_resolve_gaia_source_id_from_name_query_exception_returns_none(monkeypatch):
    class FakeSimbad:
        def add_votable_fields(self, *_args, **_kwargs):
            return None

        def query_object(self, _target_name):
            raise RuntimeError("network down")

    import astroquery.simbad as aq_simbad
    monkeypatch.setattr(aq_simbad, "Simbad", FakeSimbad, raising=False)

    out = load_gaia._resolve_gaia_source_id_from_name("HD 189733")
    assert out is None


# Tests: _resolve_source_id_from_coordinates
# Behavior: returns nearest source_id resolved from target RA/Dec cone
def test_resolve_source_id_from_coordinates_success(monkeypatch, make_global_config):
    _cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    target_name = "Target"
    right_ascension = 1.0
    declination = 2.0
    target_coord = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(load_gaia, "_get_target_coordinates", lambda _target_name, _ra, _dec: target_coord)
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _target_coord, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone)
    monkeypatch.setattr(load_gaia, "_find_target_source_id", lambda _cone_table, _target_coord: 222)

    out = load_gaia._resolve_source_id_from_coordinates(target_name, right_ascension, declination, False)
    assert out == 222


# Tests: _resolve_source_id_from_coordinates
# Behavior: raises when cone search yields no rows
def test_resolve_source_id_from_coordinates_raises_when_cone_empty(monkeypatch, make_global_config):
    _cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    target_name = "Target"
    right_ascension = 1.0
    declination = 2.0
    target_coord = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    monkeypatch.setattr(load_gaia, "_get_target_coordinates", lambda _target_name, _ra, _dec: target_coord)
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _target_coord, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: None)

    with pytest.raises(RuntimeError, match="No Gaia cone result found"):
        load_gaia._resolve_source_id_from_coordinates(target_name, right_ascension, declination, False)




# Tests: _get_target_coordinates
# Behavior: returns SkyCoord from provided RA/Dec in ICRS degrees
def test_get_target_coordinates_returns_icrs_skycoord():
    out = load_gaia._get_target_coordinates("Target", 300.1821223, 22.7097759)

    assert isinstance(out, SkyCoord)
    assert out.frame.name == "icrs"
    assert out.ra.deg == pytest.approx(300.1821223)
    assert out.dec.deg == pytest.approx(22.7097759)


# Tests: _get_target_coordinates
# Behavior: raises ValueError when RA/Dec are missing
def test_get_target_coordinates_raises_when_ra_dec_missing():
    with pytest.raises(ValueError, match="no RA/Dec available"):
        load_gaia._get_target_coordinates("Target Without Coordinates", None, None)


# Tests: lookup_target_star_gaia
# Behavior: full run-through succeeds when SIMBAD resolves source_id
def test_lookup_target_star_gaia_full_run_succeeds_with_simbad_source_id(monkeypatch, make_global_config):
    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=True)
    recorded = {"query": None, "map_called": 0, "distance_called": 0}

    def fake_query(source_id, use_async):
        recorded["query"] = (source_id, use_async)
        return _make_gaia_row(source_id=222, effective_temperature=5123.0, radius=0.88, distance=123.0)

    def fake_map(row):
        recorded["map_called"] += 1
        return {
            "source_id": int(row["source_id"]),
            "effective_temperature": float(row["effective_temperature"]),
            "radius": float(row["radius"]),
            "distance": float(row["distance"]),
        }

    def fake_distance(params):
        recorded["distance_called"] += 1
        return params

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "_query_gaia_target_star", fake_query)
    monkeypatch.setattr(load_gaia, "get_gaia_stellar_properties", fake_map)
    monkeypatch.setattr(load_gaia, "apply_distance_from_parallax_if_missing", fake_distance)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "Target Without Coordinates"},
        missing_stellar_keys=["effective_temperature", "radius", "distance"],
        cfg=cfg,
    )

    assert out["source_id"] == 222
    assert out["effective_temperature"] == 5123.0
    assert out["radius"] == 0.88
    assert out["distance"] == 123.0
    assert recorded["query"] == (222, True)
    assert recorded["map_called"] == 1
    assert recorded["distance_called"] == 1

# Tests: _find_target_source_id
# Behavior: raises ValueError for an empty cone table because argmin has no elements
def test_find_target_source_id_raises_for_empty_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([])

    with pytest.raises(ValueError, match="attempt to get argmin of an empty sequence"):
        load_gaia._find_target_source_id(table, center)

# Tests: _find_target_source_id
# Behavior: raises TypeError when cone_result is None
def test_find_target_source_id_raises_for_none_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    with pytest.raises(TypeError):
        load_gaia._find_target_source_id(None, center)


# Tests: lookup_target_star_gaia
# Behavior: full run-through fails when no source_id can be resolved
def test_lookup_target_star_gaia_full_run_raises_when_no_source_id(monkeypatch, make_global_config):
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: None)
    monkeypatch.setattr(load_gaia, "_resolve_source_id_from_coordinates", lambda _name, _ra, _dec, _async: None)

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(ValueError, match="no RA/Dec available and no Gaia source_id could be resolved from SIMBAD"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Unresolved Target"},
            missing_stellar_keys=["distance"],
            cfg=cfg,
        )


# Tests: get_gaia_stellar_properties
# Behavior: converts NaN values in Gaia fields to None
def test_get_gaia_stellar_properties_converts_nan_to_none():
    row = _make_gaia_row(effective_temperature=float("nan"), distance=float("nan"), mass=None)

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
        effective_temperature=4656.00146484375,
        radius=30.21190071105957,
        mass=4.80074405670166,
        metallicity=0.14090000092983246,
        surface_gravity=2.046299934387207,
        right_ascension=294.6925327559344,
        declination=31.24936559398053,
        distance=2401.2685546875,
        gaia_magnitude=11.795485496520996,
        parallax=0.3549924624326295,
    )

    got = load_gaia.get_gaia_stellar_properties(row, log_output=False)

    assert got["effective_temperature"] == 4656.00146484375
    assert got["radius"] == 30.21190071105957
    assert got["mass"] == 4.80074405670166
    assert got["distance"] == 2401.2685546875
    assert got["gaia_magnitude"] == 11.795485496520996


import pytest
import numpy as np


# Tests: get_gaia_stellar_properties
# Behavior: converts blank-like Gaia field values to None
@pytest.mark.parametrize(
    "field_value, metallicity_value, surface_gravity_value",
    [
        ("", np.ma.masked, np.ma.masked),
        ("   ", "   ", "   "),
    ],
)
def test_get_gaia_stellar_properties_handles_blank_like_values(field_value, metallicity_value, surface_gravity_value):
    gaia_row = _make_gaia_row(
        effective_temperature=field_value,
        distance=field_value,
        radius=field_value,
        mass=field_value,
        metallicity=metallicity_value,
        surface_gravity=surface_gravity_value,
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

    monkeypatch.setattr(load_gaia, "_run_gaia_query", fake_run)

    out = load_gaia._query_gaia_target_star(123, False)

    assert int(out["source_id"]) == 123
    assert float(out["ra"]) == 1.0
    assert recorded == {"query": "SELECT 123", "async": False}


# Tests: query_gaia
# Behavior: raises when the TAP job has no rows
def test_query_gaia_returns_empty_dict_for_empty_result(monkeypatch):
    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")
    monkeypatch.setattr(load_gaia, "_run_gaia_query", lambda query, async_flag: Table(names=("source_id",), dtype=(int,)))

    with pytest.raises(RuntimeError, match="No Gaia row returned"):
        load_gaia._query_gaia_target_star(123, False)


# Tests: query_gaia
# Behavior: propagates TAP job failures
def test_query_gaia_propagates_run_gaia_job_failure(monkeypatch):
    import pytest

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")

    def _raise(query, async_flag):
        raise RuntimeError("boom")

    monkeypatch.setattr(load_gaia, "_run_gaia_query", _raise)

    with pytest.raises(RuntimeError):
        load_gaia._query_gaia_target_star(123, False)


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


# Tests: apply_distance_from_parallax_if_missing
# Behavior: sets distance from positive parallax when distance is missing
def test_apply_distance_from_parallax_if_missing_sets_distance_from_positive_parallax():
    star_params = {"name": "Target", "parallax": 10.0}

    out = load_gaia.apply_distance_from_parallax_if_missing(star_params)

    assert out is star_params
    assert star_params == {"name": "Target", "parallax": 10.0, "distance": 100.0}


# Tests: apply_distance_from_parallax_if_missing
# Behavior: keeps existing distance unchanged
def test_apply_distance_from_parallax_if_missing_keeps_existing_distance():
    star_params = {"name": "Target", "parallax": 10.0, "distance": 42.0}

    out = load_gaia.apply_distance_from_parallax_if_missing(star_params)

    assert out is star_params
    assert star_params == {"name": "Target", "parallax": 10.0, "distance": 42.0}


# Tests: apply_distance_from_parallax_if_missing
# Behavior: leaves values unchanged for zero negative missing and invalid parallax
@pytest.mark.parametrize(
    "star_params",
    [
        {"name": "Target"},
        {"name": "Target", "parallax": None},
        {"name": "Target", "parallax": 0.0},
        {"name": "Target", "parallax": -1.0},
        {"name": "Target", "parallax": ""},
        {"name": "Target", "parallax": "   "},
        {"name": "Target", "parallax": "utter nonsense"},
        {"name": "Target", "parallax": object()},
    ],
)
def test_apply_distance_from_parallax_if_missing_rejects_bullshit_input(star_params):
    original = dict(star_params)

    out = load_gaia.apply_distance_from_parallax_if_missing(star_params)

    assert out is star_params
    assert star_params == original


# Tests: apply_distance_from_parallax_if_missing
# Behavior: accepts numeric strings for positive parallax
def test_apply_distance_from_parallax_if_missing_accepts_numeric_string():
    star_params = {"name": "Target", "parallax": "20.0"}

    out = load_gaia.apply_distance_from_parallax_if_missing(star_params)

    assert out is star_params
    assert star_params == {"name": "Target", "parallax": "20.0", "distance": 50.0}


# Tests: load_stellar pipeline (Gaia merge + parallax distance)
# Behavior: same order as load_stellar_and_planetary_properties — merge Gaia (distance None, parallax set) then parallax fallback fills distance_pc
def test_merge_gaia_then_apply_distance_fills_distance_from_parallax():
    star_params = {"name": "TOI-6038 A", "right_ascension": 51.0, "declination": 40.0}
    gaia_return = {"distance": None, "parallax": 5.61220048991683, "effective_temperature": 6070.0}

    merged = merge_gaia_into_star_params(star_params, gaia_return)

    assert merged["distance"] is None
    assert merged["parallax"] == 5.61220048991683
    assert merged["effective_temperature"] == 6070.0

    out = load_gaia.apply_distance_from_parallax_if_missing(merged)

    assert out is merged
    expected_pc = 1000.0 / 5.61220048991683
    assert out["distance"] == pytest.approx(expected_pc)
    assert out["parallax"] == 5.61220048991683



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
        g_mag_cutoff=15.0,
        GAIA_USE_ASYNC_JOBS=False,
        radius_arcsec=30.0,
    )

    assert out is None




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

    out = load_gaia._run_gaia_query("SELECT 1", use_async)
    assert len(out) == 1
    assert out["mode"][0] == expected_path

# Tests: _gaia_query_builder_for_cone_search
# Behavior: builds ADQL cone query with radius conversion and optional magnitude clause
def test_gaia_query_builder_for_cone_search_builds_expected_query():
    query = load_gaia._gaia_query_builder_for_cone_search(10.5, -20.25, 720.0, 20.0)

    assert "FROM gaiadr3.gaia_source" in query
    assert "POINT('ICRS', ra, dec)" in query
    assert "CIRCLE('ICRS', 10.5, -20.25, 0.2)" in query
    assert "AND phot_g_mean_mag < 20.0" in query

# Tests: _gaia_cone_search
# Behavior: uses query builder and _run_gaia_query instead of astroquery cone_search
def test_gaia_cone_search_uses_query_builder_and_run_query(monkeypatch):
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    recorded = {"query": None, "async_flag": None}

    monkeypatch.setattr(load_gaia, "_gaia_query_builder_for_cone_search", lambda ra_deg, dec_deg, radius_arcsec, g_mag_limit: "SELECT cone")
    monkeypatch.setattr(
        load_gaia,
        "_run_gaia_query",
        lambda query, async_flag: recorded.update({"query": query, "async_flag": async_flag}) or Table(rows=[(111, 1.0, 2.0)], names=("source_id", "ra", "dec")),
    )

    out = load_gaia._gaia_cone_search(center, radius_arcsec=6.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=True)

    assert out is not None
    assert recorded == {"query": "SELECT cone", "async_flag": True}
    assert list(out["source_id"]) == [111]

# Tests: gaia_lookup_for_background_stars
# Behavior: uses fixed catalog cutoff 20.0 for cone prequery and simulation cutoff for source_id query
def test_gaia_lookup_for_background_stars_uses_two_different_mag_cutoffs(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=1.0, declination=2.0, gaia_source_id=111)
    recorded = {"cone_limit": None, "source_ids_limit": None}

    cone = Table(rows=[(111, 1.0, 2.0), (222, 1.01, 2.0)], names=("source_id", "ra", "dec"))
    joined = Table(rows=[(222, 1.01, 2.0, 12.3)], names=("source_id", "right_ascension", "declination", "gaia_magnitude"))

    def fake_cone_search(center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS):
        recorded["cone_limit"] = g_mag_limit
        return cone

    def fake_query_for_source_ids(source_ids, g_mag_limit):
        recorded["source_ids_limit"] = g_mag_limit
        return "SELECT joined"

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", fake_cone_search)
    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_ids", fake_query_for_source_ids)
    monkeypatch.setattr(load_gaia, "_run_gaia_query", lambda query, async_flag: joined)

    out = load_gaia.gaia_lookup_for_background_stars(star, g_mag_cutoff=15.5, GAIA_USE_ASYNC_JOBS=False, radius_arcsec=30.0)

    assert out is not None
    assert recorded["cone_limit"] == 20.0
    assert recorded["source_ids_limit"] == 15.5


# Tests: gaia_lookup_for_background_stars
# Behavior: removes target source_id before joined Gaia lookup
def test_gaia_lookup_for_background_stars_removes_target_source_id_before_second_query(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=1.0, declination=2.0, gaia_source_id=111)
    cone = Table(rows=[(111, 1.0, 2.0), (222, 1.01, 2.0), (333, 1.02, 2.0)], names=("source_id", "ra", "dec"))
    recorded = {"ids": None}

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone)

    def fake_query_for_source_ids(source_ids, g_mag_limit):
        recorded["ids"] = [int(x) for x in source_ids]
        return "SELECT joined"

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_ids", fake_query_for_source_ids)
    monkeypatch.setattr(load_gaia, "_run_gaia_query", lambda query, async_flag: Table(rows=[(222,), (333,)], names=("source_id",)))

    out = load_gaia.gaia_lookup_for_background_stars(star, g_mag_cutoff=20.0, GAIA_USE_ASYNC_JOBS=False, radius_arcsec=30.0)

    assert out is not None
    assert recorded["ids"] == [222, 333]


# Tests: gaia_lookup_for_background_stars
# Behavior: returns None when only the target star is present in the cone result
def test_gaia_lookup_for_background_stars_returns_none_when_only_target_is_in_cone(monkeypatch, make_star):
    star = make_star(name="Target", right_ascension=1.0, declination=2.0, gaia_source_id=111)
    cone = Table(rows=[(111, 1.0, 2.0)], names=("source_id", "ra", "dec"))

    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda center, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone)

    out = load_gaia.gaia_lookup_for_background_stars(star, g_mag_cutoff=20.0, GAIA_USE_ASYNC_JOBS=False, radius_arcsec=30.0)

    assert out is None


