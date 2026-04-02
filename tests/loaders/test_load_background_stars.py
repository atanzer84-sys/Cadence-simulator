"""Tests for loaders.load_background_stars."""

import numpy as np
from astropy.table import Table

from loaders import load_background_stars as lbs


# Tests: load_required_stellar_parameters
# Behavior: proxies required_stellar_parameters from excel mapping
def test_load_required_stellar_parameters_reads_mapping(monkeypatch):
    monkeypatch.setattr(
        lbs,
        "load_excel_mapping",
        lambda: {"required_stellar_parameters": ["mass", "radius", "effective_temperature"]},
    )

    required = lbs.load_required_stellar_parameters()
    assert required == ["mass", "radius", "effective_temperature"]


# Tests: save_background_stars_csv
# Behavior: writes CSV with normalized name and emits info/print message
def test_save_background_stars_csv_writes_file_and_emits_message(tmp_path, caplog, capsys):
    table = Table({"ra": [10.0], "dec": [20.0], "source_id": [123]})

    with caplog.at_level("INFO"):
        lbs._save_background_stars_csv(table, tmp_path, "Target Name")

    expected_path = tmp_path / "Target_Name.csv"
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0

    expected_msg = f"Background stars: saved to {expected_path} (move to data/BackgroundStars/ for cache)"
    assert expected_msg in caplog.text
    assert expected_msg in capsys.readouterr().out


# Tests: _spectroscopy_radius_arcsec
# Behavior: computes diagonal reach from slit half-width and half-length
def test_spectroscopy_radius_arcsec_uses_slit_diagonal(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(slit_half_width_arcsec=3.0, slit_half_length_arcsec=4.0)
    assert lbs.spectroscopy_radius_arcsec(ch) == 5.0


# Tests: _photometry_radius_arcsec
# Behavior: computes diagonal reach from detector half-sizes in arcsec
def test_photometry_radius_arcsec_uses_detector_diagonal(make_photometry_channel):
    ch = make_photometry_channel(x_pixels=6, y_pixels=8, pixel_scale=1.0)
    # half-width=3, half-height=4 -> diagonal=5
    assert lbs.photometry_radius_arcsec(ch) == 5.0


# Tests: drop_stars_outside_max_radius
# Behavior: keeps only stars inside maximum channel reach radius
def test_drop_stars_outside_max_radius_filters_by_max_channel_radius(make_spectroscopy_channel, make_photometry_channel):
    table = Table({"separation_arcsec": [1.0, 5.0, 9.0]})
    nuv = make_spectroscopy_channel(slit_half_width_arcsec=3.0, slit_half_length_arcsec=4.0)  # radius 5
    nir = make_photometry_channel(x_pixels=6, y_pixels=8, pixel_scale=1.0)  # radius 5

    filtered = lbs._drop_stars_outside_max_radius(table, nuv=nuv, vis=None, nir=nir)

    assert len(filtered) == 2
    assert np.allclose(filtered["separation_arcsec"], [1.0, 5.0])


# Tests: drop_stars_outside_max_radius
# Behavior: VIS spectroscopy channel contributes to radius filter
def test_drop_stars_outside_max_radius_vis_channel_path(make_spectroscopy_channel):
    table = Table({"separation_arcsec": [4.9, 5.0, 5.1]})
    vis = make_spectroscopy_channel(slit_half_width_arcsec=3.0, slit_half_length_arcsec=4.0)  # radius 5

    filtered = lbs._drop_stars_outside_max_radius(table, nuv=None, vis=vis, nir=None)

    assert len(filtered) == 2
    assert np.allclose(filtered["separation_arcsec"], [4.9, 5.0])


# Tests: drop_stars_outside_max_radius
# Behavior: returns input unchanged when no channels are enabled
def test_drop_stars_outside_max_radius_no_channels_returns_input():
    table = Table({"separation_arcsec": [1.0, 2.0, 3.0]})
    out = lbs._drop_stars_outside_max_radius(table, nuv=None, vis=None, nir=None)
    assert len(out) == len(table)
    assert np.allclose(out["separation_arcsec"], table["separation_arcsec"])


# Tests: drop_stars_outside_max_radius
# Behavior: non-positive max radius is a no-op
def test_drop_stars_outside_max_radius_non_positive_radius_noop(make_spectroscopy_channel):
    table = Table({"separation_arcsec": [1.0, 2.0, 3.0]})
    nuv = make_spectroscopy_channel(slit_half_width_arcsec=0.0, slit_half_length_arcsec=0.0)
    out = lbs._drop_stars_outside_max_radius(table, nuv=nuv, vis=None, nir=None)
    assert len(out) == len(table)
    assert np.allclose(out["separation_arcsec"], table["separation_arcsec"])


# Tests: drop_stars_outside_max_radius
# Behavior: empty input table remains empty
def test_drop_stars_outside_max_radius_empty_table_stays_empty(make_spectroscopy_channel):
    table = Table({"separation_arcsec": []})
    nuv = make_spectroscopy_channel(slit_half_width_arcsec=3.0, slit_half_length_arcsec=4.0)
    out = lbs._drop_stars_outside_max_radius(table, nuv=nuv, vis=None, nir=None)
    assert len(out) == 0


# Tests: annotate_background_star_offsets_arcsec
# Behavior: adds relative offset and separation columns for background stars
def test_annotate_background_star_offsets_arcsec_adds_columns(make_star):
    star = make_star(right_ascension=100.0, declination=20.0)
    table = Table(
        {
            "right_ascension": [100.0, 100.001],
            "declination": [20.0, 20.0],
        }
    )

    out = lbs._annotate_background_star_offsets_arcsec(table, star)

    assert "relative_dx_arcsec" in out.colnames
    assert "relative_dy_arcsec" in out.colnames
    assert "separation_arcsec" in out.colnames
    assert out["separation_arcsec"][0] == 0.0
    assert out["separation_arcsec"][1] > 0.0


# Tests: load_background_csv_if_exists
# Behavior: returns None when cache CSV is missing
def test_load_background_csv_if_exists_missing_returns_none(tmp_path, make_star):
    star = make_star(name="Target Name")

    table = lbs._load_background_csv_if_exists(star, repo_root=tmp_path)

    assert table is None


# Tests: load_background_csv_if_exists
# Behavior: loads cached CSV from data/BackgroundStars path
def test_load_background_csv_if_exists_loads_table(tmp_path, make_star):
    bg_dir = tmp_path / "data" / "BackgroundStars"
    bg_dir.mkdir(parents=True)
    csv_path = bg_dir / "Target_Name.csv"
    csv_path.write_text("right_ascension,declination,source_id\n10.0,20.0,123\n", encoding="utf-8")
    star = make_star(name="Target Name")

    table = lbs._load_background_csv_if_exists(star, repo_root=tmp_path)

    assert table is not None
    assert len(table) == 1
    assert float(table["right_ascension"][0]) == 10.0


# Tests: load_background_csv_if_exists
# Behavior: malformed CSV raises parser error instead of silently succeeding
def test_load_background_csv_if_exists_malformed_csv_raises(tmp_path, make_star, monkeypatch):
    import pytest

    bg_dir = tmp_path / "data" / "BackgroundStars"
    bg_dir.mkdir(parents=True)
    csv_path = bg_dir / "Target_Name.csv"
    csv_path.write_text("right_ascension,declination,source_id\n10.0,20.0,123\n", encoding="utf-8")
    star = make_star(name="Target Name")

    def _raise_malformed(*args, **kwargs):
        raise ValueError("Malformed CSV")

    monkeypatch.setattr(lbs.Table, "read", staticmethod(_raise_malformed))
    with pytest.raises(ValueError):
        lbs._load_background_csv_if_exists(star, repo_root=tmp_path)


# Tests: load_background_csv_if_exists
# Behavior: missing required ra/dec columns raises clear ValueError
def test_load_background_csv_if_exists_missing_required_columns_raises(tmp_path, make_star):
    import pytest

    bg_dir = tmp_path / "data" / "BackgroundStars"
    bg_dir.mkdir(parents=True)
    csv_path = bg_dir / "Target_Name.csv"
    csv_path.write_text("source_id\n123\n", encoding="utf-8")
    star = make_star(name="Target Name")

    with pytest.raises(ValueError):
        lbs._load_background_csv_if_exists(star, repo_root=tmp_path)


# Tests: annotate_background_star_offsets_arcsec
# Behavior: missing required coordinate columns raises KeyError
def test_annotate_background_star_offsets_arcsec_missing_columns_raises(make_star):
    import pytest

    star = make_star(right_ascension=100.0, declination=20.0)
    table = Table({"source_id": [1]})

    with pytest.raises(KeyError):
        lbs._annotate_background_star_offsets_arcsec(table, star)


# Tests: _apply_background_star_magnitude_cutoff
# Behavior: keeps rows <= max_mag; missing magnitude column skips filter
def test_apply_background_star_magnitude_cutoff_behavior():
    table = Table({"gaia_magnitude": [10.0, 12.0], "source_id": [1, 2]})
    out = lbs._apply_background_star_magnitude_cutoff(table, max_mag=11.0)
    assert len(out) == 1
    assert float(out["gaia_magnitude"][0]) == 10.0

    no_mag = Table({"source_id": [1, 2]})
    unchanged = lbs._apply_background_star_magnitude_cutoff(no_mag, max_mag=11.0)
    assert len(unchanged) == 2


# Tests: _set_background_star_name
# Behavior: uses source_id when present, falls back to '0000' when missing
def test_set_background_star_name_from_source_id_or_fallback():
    star_params = {}
    row_with_id = Table({"source_id": [123456789]} )[0]
    lbs._set_background_star_name(star_params, row_with_id)
    assert star_params["name"] == "gaia_123456789"

    star_params = {}
    row_without_id = Table({"ra": [1.0]})[0]
    lbs._set_background_star_name(star_params, row_without_id)
    assert star_params["name"] == "0000"


# Tests: _set_background_star_name
# Behavior: explicit None source_id falls back to '0000'
def test_set_background_star_name_source_id_none_falls_back():
    star_params = {}
    row_with_none = Table({"source_id": [None]})[0]
    lbs._set_background_star_name(star_params, row_with_none)
    assert star_params["name"] == "0000"


# Tests: _ensure_required_properties
# Behavior: returns False when required properties are missing, True otherwise without mutating keys
def test_ensure_required_properties_handles_missing_and_present(monkeypatch):
    monkeypatch.setattr(lbs, "get_missing_properties", lambda star_params, required_keys, log_output=False: ["radius"])
    ok = lbs._ensure_required_properties({"name": "bg1"}, required_keys=["radius"])
    assert ok is False

    monkeypatch.setattr(lbs, "get_missing_properties", lambda star_params, required_keys, log_output=False: [])
    star_params = {"name": "bg2", "distance": 10.0}
    ok = lbs._ensure_required_properties(star_params, required_keys=["distance"])
    assert ok is True
    assert "distance" in star_params




# Tests: create_background_star_catalog
# Behavior: missing offset columns raises KeyError at offset extraction
def test_create_background_star_catalog_missing_offset_columns_raises(monkeypatch, make_global_config):
    import pytest

    cfg = make_global_config(magnitude_cutoff=15.0)
    table = Table({"source_id": [123]})

    monkeypatch.setattr(lbs, "load_required_stellar_parameters", lambda: ["name", "gaia_magnitude"])
    monkeypatch.setattr(lbs, "get_gaia_stellar_properties", lambda row, log_output=False: {"gaia_magnitude": 10.0})
    monkeypatch.setattr(lbs, "apply_distance_from_parallax_if_missing", lambda p: p)
    monkeypatch.setattr(lbs, "apply_radius_from_teff_mag_distance_if_missing", lambda p: p)
    monkeypatch.setattr(lbs, "infer_mamajek", lambda p, log_output=False: p)
    monkeypatch.setattr(lbs, "apply_log_r", lambda p, _cfg, log_output=False: p)
    monkeypatch.setattr(lbs, "_ensure_required_properties", lambda p, req: True)

    class _FakeStar:
        def __init__(self, name):
            self.name = name
            self.mass = 1.0
            self.gaia_magnitude = 10.0
            self.effective_temperature = 5500.0
            self.radius = 1.0
            self.right_ascension = 1.0
            self.declination = 2.0
            self.distance_pc = 10.0

    monkeypatch.setattr(lbs.Star, "from_params", staticmethod(lambda params, required, log_output=False: _FakeStar(params["name"])))

    with pytest.raises(KeyError):
        lbs.create_background_star_catalog(table, cfg)


# Tests: create_background_star_catalog
# Behavior: runs infer_mamajek and apply_log_r only when guarded keys exist
def test_create_background_star_catalog_teff_and_radius_guards(monkeypatch, make_global_config):
    cfg = make_global_config(magnitude_cutoff=15.0)
    table = Table(
        {
            "source_id": [101, 202],
            "relative_dx_arcsec": [0.1, 0.2],
            "relative_dy_arcsec": [0.3, 0.4],
        }
    )
    calls = {"infer": 0, "logr": 0}

    monkeypatch.setattr(lbs, "load_required_stellar_parameters", lambda: ["name", "gaia_magnitude"])
    monkeypatch.setattr(
        lbs,
        "get_gaia_stellar_properties",
        lambda row, log_output=False: (
            {"gaia_magnitude": 10.0, "effective_temperature": 5500.0, "radius": 1.2}
            if int(row["source_id"]) == 101
            else {"gaia_magnitude": 10.0, "effective_temperature": None, "radius": None}
        ),
    )
    monkeypatch.setattr(lbs, "apply_distance_from_parallax_if_missing", lambda p: p)
    monkeypatch.setattr(lbs, "apply_radius_from_teff_mag_distance_if_missing", lambda p: p)

    def _infer(params, log_output=False):
        calls["infer"] += 1
        return params

    def _logr(params, _cfg, log_output=False):
        calls["logr"] += 1
        return params

    monkeypatch.setattr(lbs, "infer_mamajek", _infer)
    monkeypatch.setattr(lbs, "apply_log_r", _logr)
    monkeypatch.setattr(lbs, "_ensure_required_properties", lambda p, req: True)

    class _FakeStar:
        def __init__(self, name):
            self.name = name
            self.mass = 1.0
            self.gaia_magnitude = 10.0
            self.effective_temperature = 5500.0
            self.radius = 1.0
            self.right_ascension = 1.0
            self.declination = 2.0
            self.distance_pc = 10.0

    monkeypatch.setattr(lbs.Star, "from_params", staticmethod(lambda params, required, log_output=False: _FakeStar(params["name"])))

    catalog = lbs.create_background_star_catalog(table, cfg)

    assert len(catalog.stars_by_id) == 2
    assert calls["infer"] == 1
    assert calls["logr"] == 1


# Tests: create_background_star_catalog
# Behavior: when radius exists, apply_log_r is invoked with cfg and log_output=False
def test_create_background_star_catalog_radius_branch_calls_log_r_fallback(monkeypatch, make_global_config):
    cfg = make_global_config(magnitude_cutoff=15.0)
    table = Table(
        {
            "source_id": [303],
            "relative_dx_arcsec": [0.0],
            "relative_dy_arcsec": [0.0],
        }
    )
    seen = {"called": 0, "cfg": None, "log_output": None}

    monkeypatch.setattr(lbs, "load_required_stellar_parameters", lambda: ["name", "gaia_magnitude"])
    monkeypatch.setattr(lbs, "get_gaia_stellar_properties", lambda row, log_output=False: {"gaia_magnitude": 10.0, "radius": 1.1})
    monkeypatch.setattr(lbs, "apply_distance_from_parallax_if_missing", lambda p: p)
    monkeypatch.setattr(lbs, "apply_radius_from_teff_mag_distance_if_missing", lambda p: p)
    monkeypatch.setattr(lbs, "infer_mamajek", lambda p, log_output=False: p)

    def _apply_log_r(params, cfg_in, log_output=False):
        seen["called"] += 1
        seen["cfg"] = cfg_in
        seen["log_output"] = log_output
        return params

    monkeypatch.setattr(lbs, "apply_log_r", _apply_log_r)
    monkeypatch.setattr(lbs, "_ensure_required_properties", lambda p, req: True)

    class _FakeStar:
        def __init__(self, name):
            self.name = name
            self.mass = 1.0
            self.gaia_magnitude = 10.0
            self.effective_temperature = 5500.0
            self.radius = 1.0
            self.right_ascension = 1.0
            self.declination = 2.0
            self.distance_pc = 10.0

    monkeypatch.setattr(lbs.Star, "from_params", staticmethod(lambda params, required, log_output=False: _FakeStar(params["name"])))

    catalog = lbs.create_background_star_catalog(table, cfg)

    assert len(catalog.stars_by_id) == 1
    assert seen["called"] == 1
    assert seen["cfg"] is cfg
    assert seen["log_output"] is False


# Tests: lookup_background_stars
# Behavior: CSV cache hit skips Gaia and does not save CSV
def test_lookup_background_stars_uses_csv_cache_path(monkeypatch, make_star, make_global_config, make_run_context):
    star = make_star(name="Target")
    cfg = make_global_config(magnitude_cutoff=15.0, GAIA_USE_ASYNC_JOBS=False, gaia_conesearch_radius_arcsec=30.0)
    ctx = make_run_context()
    cached = Table({"right_ascension": [1.0], "declination": [2.0], "source_id": [123]})
    catalog_obj = lbs.StarCatalog()

    monkeypatch.setattr(lbs, "get_global_config", lambda: cfg)
    monkeypatch.setattr(lbs, "_load_background_csv_if_exists", lambda s: cached)
    monkeypatch.setattr(lbs, "gaia_lookup_for_background_stars", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gaia should not be called")))
    monkeypatch.setattr(lbs, "_save_background_stars_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("save should not be called")))
    monkeypatch.setattr(lbs, "_annotate_background_star_offsets_arcsec", lambda table, target: table)
    monkeypatch.setattr(lbs, "_drop_stars_outside_max_radius", lambda table, nuv, vis, nir: table)
    monkeypatch.setattr(lbs, "create_background_star_catalog", lambda table, cfg_in: catalog_obj)

    out = lbs.lookup_background_stars(None, None, None, ctx, star)
    assert out is catalog_obj


# Tests: lookup_background_stars
# Behavior: missing CSV triggers Gaia query and saves non-empty Gaia result
def test_lookup_background_stars_queries_gaia_and_saves_when_rows(monkeypatch, make_star, make_global_config, make_run_context):
    star = make_star(name="Target")
    cfg = make_global_config(magnitude_cutoff=16.0, GAIA_USE_ASYNC_JOBS=True, gaia_conesearch_radius_arcsec=77.0)
    ctx = make_run_context()
    gaia_table = Table({"right_ascension": [1.0], "declination": [2.0], "source_id": [999]})
    calls = {"save": 0}
    catalog_obj = lbs.StarCatalog()

    monkeypatch.setattr(lbs, "get_global_config", lambda: cfg)
    monkeypatch.setattr(lbs, "_load_background_csv_if_exists", lambda s: None)
    monkeypatch.setattr(
        lbs,
        "gaia_lookup_for_background_stars",
        lambda s, g_mag_cutoff, GAIA_USE_ASYNC_JOBS, radius_arcsec: gaia_table
        if (g_mag_cutoff, GAIA_USE_ASYNC_JOBS, radius_arcsec) == (20.0, True, 77.0)
        else (_ for _ in ()).throw(AssertionError("unexpected gaia args")),
    )
    monkeypatch.setattr(lbs, "_save_background_stars_csv", lambda table, output_dir, star_name: calls.__setitem__("save", calls["save"] + 1))
    monkeypatch.setattr(lbs, "_annotate_background_star_offsets_arcsec", lambda table, target: table)
    monkeypatch.setattr(lbs, "_drop_stars_outside_max_radius", lambda table, nuv, vis, nir: table)
    monkeypatch.setattr(lbs, "create_background_star_catalog", lambda table, cfg_in: catalog_obj)

    out = lbs.lookup_background_stars(None, None, None, ctx, star)
    assert out is catalog_obj
    assert calls["save"] == 1


# Tests: lookup_background_stars
# Behavior: empty Gaia result returns empty catalog immediately
def test_lookup_background_stars_empty_result_returns_empty_catalog(monkeypatch, make_star, make_global_config, make_run_context):
    star = make_star(name="Target")
    cfg = make_global_config(magnitude_cutoff=15.0, GAIA_USE_ASYNC_JOBS=False, gaia_conesearch_radius_arcsec=30.0)
    ctx = make_run_context()
    calls = {"annotate": 0, "drop": 0, "create": 0}

    monkeypatch.setattr(lbs, "get_global_config", lambda: cfg)
    monkeypatch.setattr(lbs, "_load_background_csv_if_exists", lambda s: None)
    monkeypatch.setattr(lbs, "gaia_lookup_for_background_stars", lambda *args, **kwargs: Table({"right_ascension": [], "declination": [], "source_id": []}))
    monkeypatch.setattr(lbs, "_save_background_stars_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(lbs, "_annotate_background_star_offsets_arcsec", lambda *args, **kwargs: calls.__setitem__("annotate", calls["annotate"] + 1))
    monkeypatch.setattr(lbs, "_drop_stars_outside_max_radius", lambda *args, **kwargs: calls.__setitem__("drop", calls["drop"] + 1))
    monkeypatch.setattr(lbs, "create_background_star_catalog", lambda *args, **kwargs: calls.__setitem__("create", calls["create"] + 1))

    out = lbs.lookup_background_stars(None, None, None, ctx, star)
    assert isinstance(out, lbs.StarCatalog)
    assert len(out.stars_by_id) == 0
    assert calls == {"annotate": 0, "drop": 0, "create": 0}
