import numpy as np


class FakeConeJob:
    def __init__(self, results):
        self._results = results

    def get_results(self):
        return self._results


class FakeQueryJob:
    def __init__(self, results):
        self._results = results

    def get_results(self):
        return self._results


class FakeGaiaTable:
    """
    Minimal table compatible with:
      len(table)
      table["phot_g_mean_mag"] -> numpy array
      table[idx]["source_id"] -> value
    """
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return np.array([row[key] for row in self._rows], dtype=float)
        return self._rows[key]


class FakeResultTable:
    """
    Minimal table compatible with:
      len(table)
      table.colnames
      table[0][col]
    """
    def __init__(self, rows):
        self._rows = rows
        self.colnames = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def test_lookup_star_gaia_returns_only_missing_keys(monkeypatch):
    from loaders import gaia_lookup

    # prevent network coordinate resolution
    monkeypatch.setattr(gaia_lookup.SkyCoord, "from_name", lambda _name: object())

    cone_rows = [
        {"phot_g_mean_mag": 12.0, "source_id": 111},
        {"phot_g_mean_mag": 10.0, "source_id": 222},
        {"phot_g_mean_mag": 11.0, "source_id": 333},
    ]
    cone_table = FakeGaiaTable(cone_rows)

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
    result_table = FakeResultTable([gaia_row])

    class FakeGaia:
        @staticmethod
        def cone_search_async(coord, radius):
            return FakeConeJob(cone_table)

        @staticmethod
        def launch_job_async(query):
            return FakeQueryJob(result_table)

    monkeypatch.setattr(gaia_lookup, "Gaia", FakeGaia, raising=True)

    star_params = {"name": "HD 202772 A"}
    missing = ["effective_temperature", "radius", "log_r"]  # log_r exists but is None by design

    out = gaia_lookup.lookup_star_gaia(star_params, missing_star=missing)

    assert out == {
        "effective_temperature": 5777.0,
        "radius": 1.01,
        "log_r": None,
    }


def test_lookup_star_gaia_returns_empty_dict_when_cone_search_empty(monkeypatch):
    from loaders import gaia_lookup

    monkeypatch.setattr(gaia_lookup.SkyCoord, "from_name", lambda _name: object())

    class FakeGaia:
        @staticmethod
        def cone_search_async(coord, radius):
            return FakeConeJob(FakeGaiaTable([]))

        @staticmethod
        def launch_job_async(query):
            raise AssertionError("launch_job_async must not be called when no match exists")

    monkeypatch.setattr(gaia_lookup, "Gaia", FakeGaia, raising=True)

    out = gaia_lookup.lookup_star_gaia({"name": "No Match"}, missing_star=["effective_temperature"])
    assert out == {}


def test_lookup_star_gaia_returns_empty_dict_on_any_exception(monkeypatch):
    from loaders import gaia_lookup

    monkeypatch.setattr(gaia_lookup.SkyCoord, "from_name", lambda _name: object())

    class FakeGaia:
        @staticmethod
        def cone_search_async(coord, radius):
            raise RuntimeError("gaia down")

    monkeypatch.setattr(gaia_lookup, "Gaia", FakeGaia, raising=True)

    out = gaia_lookup.lookup_star_gaia({"name": "HD 202772 A"}, missing_star=["effective_temperature"])
    assert out == {}


def test_get_gaia_stellar_properties_converts_nan_to_none():
    from loaders.gaia_lookup import get_gaia_stellar_properties

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

    out = get_gaia_stellar_properties(row)

    assert out["effective_temperature"] is None
    assert out["radius"] == 1.0
    assert out["mass"] is None
    assert out["distance"] is None
    assert out["gaia_magnitude"] == 10.0
    assert "log_r" in out
