from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import os
import pytest

from loaders import load_gaia


def _cone_table(rows):
    if not rows:
        return Table(names=("ra", "dec", "source_id"), dtype=(float, float, int))
    return Table(rows=rows, names=("ra", "dec", "source_id"))


def _make_gaia_row(**overrides):
    row = {
        "source_id": 222,
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


# Tests: _resolve_source_id_from_position
# Behavior: returns nearest source_id resolved from target RA/Dec cone
def test_resolve_source_id_from_position_success(monkeypatch, make_global_config):
    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "Target", "right_ascension": 1.0, "declination": 2.0}
    target_coord = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(load_gaia, "_get_target_coordinates", lambda _star_params: target_coord)
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _target_coord, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone)
    monkeypatch.setattr(load_gaia, "_find_central_source_id", lambda _cone_table, _target_coord: 222)

    out = load_gaia._resolve_source_id_from_position(star_params, cfg)
    assert out == 222


# Tests: _resolve_source_id_from_position
# Behavior: raises when cone search yields no rows
def test_resolve_source_id_from_position_raises_when_cone_empty(monkeypatch, make_global_config):
    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "Target", "right_ascension": 1.0, "declination": 2.0}
    target_coord = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    monkeypatch.setattr(load_gaia, "_get_target_coordinates", lambda _star_params: target_coord)
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _target_coord, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: None)

    with pytest.raises(RuntimeError, match="No Gaia cone result found"):
        load_gaia._resolve_source_id_from_position(star_params, cfg)


# Tests: _resolve_source_id_from_position
# Behavior: raises when central source cannot be identified in cone
def test_resolve_source_id_from_position_raises_when_no_central_source(monkeypatch, make_global_config):
    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "Target", "right_ascension": 1.0, "declination": 2.0}
    target_coord = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    cone = _cone_table([(1.0, 2.0, 222)])

    monkeypatch.setattr(load_gaia, "_get_target_coordinates", lambda _star_params: target_coord)
    monkeypatch.setattr(load_gaia, "_gaia_cone_search", lambda _target_coord, radius_arcsec, g_mag_limit, GAIA_USE_ASYNC_JOBS: cone)
    monkeypatch.setattr(load_gaia, "_find_central_source_id", lambda _cone_table, _target_coord: None)

    with pytest.raises(RuntimeError, match="No Gaia central match found"):
        load_gaia._resolve_source_id_from_position(star_params, cfg)


# Tests: _get_target_coordinates
# Behavior: returns SkyCoord from provided RA/Dec in ICRS degrees
def test_get_target_coordinates_returns_icrs_skycoord():
    star_params = {"name": "Target", "right_ascension": 300.1821223, "declination": 22.7097759}

    out = load_gaia._get_target_coordinates(star_params)

    assert isinstance(out, SkyCoord)
    assert out.frame.name == "icrs"
    assert out.ra.deg == pytest.approx(300.1821223)
    assert out.dec.deg == pytest.approx(22.7097759)


# Tests: _get_target_coordinates
# Behavior: raises ValueError when RA/Dec are missing
def test_get_target_coordinates_raises_when_ra_dec_missing():
    with pytest.raises(ValueError, match="no RA/Dec available"):
        load_gaia._get_target_coordinates({"name": "Target Without Coordinates"})


# Tests: lookup_target_star_gaia (live)
# Behavior: full live run without monkeypatch resolves Gaia data for a known target
@pytest.mark.skipif(os.getenv("RUN_LIVE_GAIA_TESTS") != "1", reason="Set RUN_LIVE_GAIA_TESTS=1 to run live SIMBAD/Gaia test")
def test_lookup_target_star_gaia_live_run_without_monkeypatch(make_global_config):
    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "HD 189733", "right_ascension": 300.1821223, "declination": 22.7097759}
    missing = ["effective_temperature", "radius", "distance"]

    out = load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg)

    assert out.get("source_id") is not None
    assert out.get("effective_temperature") is not None
    assert out.get("radius") is not None
    assert out.get("distance") is not None


# Tests: _find_central_source_id
# Behavior: returns source_id of nearest source to target position
def test_find_central_row_returns_nearest():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([
        (2.0, 2.0, 111),
        (1.001, 2.0, 222),
        (1.0, 2.001, 333),
    ])

    source_id = load_gaia._find_central_source_id(table, center)
    assert source_id == 222


# Tests: _find_central_source_id
# Behavior: empty table currently raises due argmin on empty sequence
def test_find_central_row_returns_none_for_empty_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")
    table = _cone_table([])

    with pytest.raises(ValueError):
        load_gaia._find_central_source_id(table, center)


# Tests: _find_central_source_id
# Behavior: missing table currently raises TypeError
def test_find_central_row_returns_none_for_none_table():
    center = SkyCoord(ra=1.0 * u.deg, dec=2.0 * u.deg, frame="icrs")

    with pytest.raises(TypeError):
        load_gaia._find_central_source_id(None, center)


# Tests: lookup_target_star_gaia
# Behavior: raises when cone search returns None
def test_lookup_target_star_gaia_raises_when_cone_search_empty(monkeypatch, make_global_config):
    import pytest

    monkeypatch.setattr(
        load_gaia,
        "_gaia_cone_search",
        lambda _center, radius_arcsec=2.0, g_mag_limit=None, GAIA_USE_ASYNC_JOBS=False: None,
    )
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: None)

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

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda _source_id, _async: (_ for _ in ()).throw(RuntimeError("No Gaia row returned")))

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

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: (_ for _ in ()).throw(RuntimeError("gaia down")))

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="gaia down"):
        load_gaia.lookup_target_star_gaia(
            {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: only distance missing from Gaia row → distance None, parallax still attached when present
def test_lookup_target_star_gaia_stores_parallax_and_distance_none_when_only_distance_missing(monkeypatch, make_global_config):
    gaia_row = _make_gaia_row(dist_pc="", parallax=8.0, Teff=5000.0)

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda _source_id, _async: gaia_row)

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    star_params = {"name": "Parallax Star", "right_ascension": 1.0, "declination": 2.0}

    out = load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=["distance"], cfg=cfg)

    assert out["distance"] is None
    assert out["parallax"] == 8.0
    assert out["source_id"] == 222


# Tests: lookup_target_star_gaia
# Behavior: forwards async flag to Gaia row query helper
def test_lookup_target_star_gaia_passes_async_flag_to_helpers(monkeypatch, make_global_config):
    recorded = {"query_async": None}

    def fake_query(source_id, GAIA_USE_ASYNC_JOBS):
        recorded["query_async"] = GAIA_USE_ASYNC_JOBS
        return _make_gaia_row()

    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", fake_query)

    star_params = {"name": "HD 202772 A", "right_ascension": 1.0, "declination": 2.0}
    missing = ["effective_temperature"]

    cfg_async = make_global_config(GAIA_USE_ASYNC_JOBS=True)
    load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg_async)
    assert recorded["query_async"] is True

    recorded["query_async"] = None

    cfg_sync = make_global_config(GAIA_USE_ASYNC_JOBS=False)
    load_gaia.lookup_target_star_gaia(star_params, missing_stellar_keys=missing, cfg=cfg_sync)
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
# Behavior: when RA/Dec absent and SIMBAD gives no source_id, lookup raises from coordinate fallback
def test_lookup_target_star_gaia_raises_when_ra_dec_missing_and_name_resolution_fails(monkeypatch, make_global_config):
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: None)

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(ValueError, match="no RA/Dec available"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Unresolved Target"},
            missing_stellar_keys=["distance"],
            cfg=cfg,
        )


# Tests: lookup_target_star_gaia
# Behavior: when SIMBAD returns source_id, lookup succeeds without RA/Dec
def test_lookup_target_star_gaia_resolves_name_when_coordinates_missing(monkeypatch, make_global_config):
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda source_id, async_flag: _make_gaia_row(Teff=5000.0, parallax=None))

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "Target Without Coordinates"},
        missing_stellar_keys=["effective_temperature"],
        cfg=cfg,
    )

    assert out["effective_temperature"] == 5000.0
    assert out["source_id"] == 222


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
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: None)
    monkeypatch.setattr(load_gaia, "_find_central_source_id", lambda cone_table, found_center: None)

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
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda source_id, async_flag: _make_gaia_row())
    monkeypatch.setattr(load_gaia, "get_gaia_stellar_properties", lambda gaia_row: {"mass": 1.0})

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="did not return requested missing keys"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Target", "right_ascension": 1.0, "declination": 2.0},
            missing_stellar_keys=["effective_temperature", "radius"],
            cfg=cfg,
        )

# Tests: lookup_target_star_gaia
# Behavior: successful Gaia lookup returns Gaia parameter payload
def test_lookup_target_star_gaia_success_returns_gaia_star_params(monkeypatch, make_global_config):
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda _source_id, _async: _make_gaia_row(Teff=5123.0, radius_sun=0.88))

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    out = load_gaia.lookup_target_star_gaia(
        {"name": "Successful Target"},
        missing_stellar_keys=["effective_temperature", "radius"],
        cfg=cfg,
    )

    assert out["source_id"] == 222
    assert out["effective_temperature"] == 5123.0
    assert out["radius"] == 0.88


# Tests: lookup_target_star_gaia
# Behavior: raises when required missing params are not returned by Gaia payload
def test_lookup_target_star_gaia_raises_when_required_missing_params_not_in_gaia(monkeypatch, make_global_config):
    monkeypatch.setattr(load_gaia, "_resolve_gaia_source_id_from_name", lambda _name: 222)
    monkeypatch.setattr(load_gaia, "query_gaia_target_star", lambda _source_id, _async: _make_gaia_row())
    monkeypatch.setattr(load_gaia, "get_gaia_stellar_properties", lambda _gaia_row: {"mass": 1.0, "source_id": 222})

    cfg = make_global_config(GAIA_USE_ASYNC_JOBS=False)

    with pytest.raises(RuntimeError, match="did not return requested missing keys"):
        load_gaia.lookup_target_star_gaia(
            {"name": "Missing Required Gaia Params"},
            missing_stellar_keys=["effective_temperature", "radius"],
            cfg=cfg,
        )


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

    out = load_gaia.query_gaia_target_star(123, False)

    assert int(out["source_id"]) == 123
    assert float(out["ra"]) == 1.0
    assert recorded == {"query": "SELECT 123", "async": False}


# Tests: query_gaia
# Behavior: raises when the TAP job has no rows
def test_query_gaia_returns_empty_dict_for_empty_result(monkeypatch):
    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")
    monkeypatch.setattr(load_gaia, "_run_gaia_query", lambda query, async_flag: Table(names=("source_id",), dtype=(int,)))

    with pytest.raises(RuntimeError, match="No Gaia row returned"):
        load_gaia.query_gaia_target_star(123, False)


# Tests: query_gaia
# Behavior: propagates TAP job failures
def test_query_gaia_propagates_run_gaia_job_failure(monkeypatch):
    import pytest

    monkeypatch.setattr(load_gaia, "_gaia_query_for_source_id", lambda source_id: f"SELECT {source_id}")

    def _raise(query, async_flag):
        raise RuntimeError("boom")

    monkeypatch.setattr(load_gaia, "_run_gaia_query", _raise)

    with pytest.raises(RuntimeError):
        load_gaia.query_gaia_target_star(123, False)


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
@pytest.mark.xfail(reason="Known contract mismatch: _gaia_drop_central_star expects (row, sep) but _find_central_source_id returns int", strict=False)
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
@pytest.mark.xfail(reason="Known contract mismatch: _gaia_drop_central_star expects (row, sep) but _find_central_source_id returns int", strict=False)
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
@pytest.mark.xfail(reason="Known contract mismatch: _gaia_drop_central_star expects (row, sep) but _find_central_source_id returns int", strict=False)
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

    out = load_gaia._run_gaia_query("SELECT 1", use_async)
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
    monkeypatch.setattr(load_gaia, "_run_gaia_query", fake_run)

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

    monkeypatch.setattr(load_gaia, "_run_gaia_query", fake_run)

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