import pytest
import numpy as np


# StarCatalog.add_star / StarCatalog.get_star
# Behavior: storing a star and retrieving it returns the same object
def test_star_catalog_add_and_get_star(make_star_catalog, make_star):
    catalog = make_star_catalog()
    star = make_star()
    catalog.add_star("s1", star)
    assert catalog.get_star("s1") is star


# StarCatalog.get_star
# Behavior: requesting a missing star ID raises KeyError
def test_star_catalog_get_missing_star_raises(make_star_catalog):
    catalog = make_star_catalog()
    with pytest.raises(KeyError):
        catalog.get_star("missing")


# StarCatalog.set_offset_arcsec / StarCatalog.get_offset_arcsec
# Behavior: offsets are stored and returned as floats
def test_star_catalog_set_and_get_offset(make_star_catalog):
    catalog = make_star_catalog()
    catalog.set_offset_arcsec("s1", 1, 2)
    assert catalog.get_offset_arcsec("s1") == (1.0, 2.0)


# StarCatalog.set_offset_arcsec
# Behavior: numeric strings are accepted because float("1") works
def test_offset_accepts_string_numbers(make_star_catalog):
    catalog = make_star_catalog()
    catalog.set_offset_arcsec("s1", "1", "2")
    assert catalog.get_offset_arcsec("s1") == (1.0, 2.0)


# StarCatalog.set_offset_arcsec
# Behavior: non-numeric strings fail because float("abc") raises
def test_offset_rejects_non_numeric_strings(make_star_catalog):
    catalog = make_star_catalog()
    with pytest.raises((TypeError, ValueError)):
        catalog.set_offset_arcsec("s1", "abc", 1)
    with pytest.raises((TypeError, ValueError)):
        catalog.set_offset_arcsec("s1", 1, "xyz")


# StarCatalog.set_offset_arcsec
# Behavior: None fails because float(None) raises
def test_offset_rejects_none(make_star_catalog):
    catalog = make_star_catalog()
    with pytest.raises((TypeError, ValueError)):
        catalog.set_offset_arcsec("s1", None, 1)
    with pytest.raises((TypeError, ValueError)):
        catalog.set_offset_arcsec("s1", 1, None)


# StarCatalog.get_offset_arcsec
# Behavior: requesting offset for unknown ID raises KeyError
def test_star_catalog_get_missing_offset_raises(make_star_catalog):
    catalog = make_star_catalog()
    with pytest.raises(KeyError):
        catalog.get_offset_arcsec("missing")


# StarCatalog.counts_by_id_and_band (direct dict access)
# Behavior: storing and retrieving values works exactly as a dict
def test_counts_storage(make_star_catalog):
    arr = np.array([1, 2, 3])
    catalog = make_star_catalog(counts={("s1", "V"): arr})
    assert np.array_equal(catalog.counts_by_id_and_band[("s1", "V")], arr)


# StarCatalog.counts_by_id_and_band (direct dict access)
# Behavior: missing key raises KeyError (normal dict behavior)
def test_counts_missing_key_raises(make_star_catalog):
    catalog = make_star_catalog()
    with pytest.raises(KeyError):
        _ = catalog.counts_by_id_and_band[("missing", "V")]


# StarCatalog.counts_by_id_and_band (direct dict access)
# Behavior: stored arrays are referenced, not copied
def test_counts_reference_semantics(make_star_catalog):
    arr = np.array([1, 2, 3])
    catalog = make_star_catalog(counts={("s1", "V"): arr})
    arr[0] = 99
    assert catalog.counts_by_id_and_band[("s1", "V")][0] == 99