import numpy as np
from unittest.mock import patch

from instrument.background_star_counts import populate_background_star_counts


# Tests: populate_background_star_counts
# Behavior: returns catalog unchanged when no background stars are present
def test_populate_background_star_counts_no_stars(make_star_catalog, make_run_context, make_spectroscopy_channel):
    catalog = make_star_catalog()
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range") as prepare_mock, \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel") as compute_mock:
        result = populate_background_star_counts(catalog, nuv=nuv, vis=None, nir=None, ctx=ctx)

    # Functional contract: when there are no background stars, the catalog remains unchanged.
    # We intentionally don't assert object identity to avoid coupling to return-by-reference vs copy semantics.
    assert result.counts_by_id_and_band == {}
    assert prepare_mock.call_count == 0
    assert compute_mock.call_count == 0


# Tests: populate_background_star_counts
# Behavior: stores float32 per pixel counts for spectroscopy channels
def test_populate_background_star_counts_stores_spectroscopy_array(make_star_catalog, make_star, make_spectroscopy_channel, make_run_context):
    star = make_star(gaia_magnitude=12.0)
    catalog = make_star_catalog(stars={"star_1": star})
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")
    counts_s_px = np.array([1.0, 2.0, 3.0], dtype=float)

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range", return_value=(np.array([10.0, 20.0]), np.array([1000.0, 1100.0]))), \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel", return_value=counts_s_px):
        result = populate_background_star_counts(catalog, nuv=nuv, vis=None, nir=None, ctx=ctx)

    key = ("star_1", nuv.channel_name)
    assert key in result.counts_by_id_and_band
    assert isinstance(result.counts_by_id_and_band[key], np.ndarray)
    assert result.counts_by_id_and_band[key].dtype == np.float32
    np.testing.assert_allclose(result.counts_by_id_and_band[key], counts_s_px.astype(np.float32))


# Tests: populate_background_star_counts
# Behavior: stores summed scalar counts for photometry channels
def test_populate_background_star_counts_stores_photometry_scalar(make_star_catalog, make_star, make_photometry_channel, make_run_context):
    star = make_star(gaia_magnitude=12.0)
    catalog = make_star_catalog(stars={"star_1": star})
    ctx = make_run_context()
    nir = make_photometry_channel(channel_name="NIR")
    counts_s_px = np.array([1.0, 2.0, 3.0], dtype=float)

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range", return_value=(np.array([10.0, 20.0]), np.array([1000.0, 1100.0]))), \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel", return_value=counts_s_px):
        result = populate_background_star_counts(catalog, nuv=None, vis=None, nir=nir, ctx=ctx)

    key = ("star_1", nir.channel_name)
    assert key in result.counts_by_id_and_band
    assert isinstance(result.counts_by_id_and_band[key], float)
    assert result.counts_by_id_and_band[key] == float(np.sum(counts_s_px))


# Tests: populate_background_star_counts
# Behavior: stores counts for all enabled channels
def test_populate_background_star_counts_processes_multiple_channels(make_star_catalog, make_star, make_spectroscopy_channel, make_photometry_channel, make_run_context):
    star = make_star(gaia_magnitude=12.0)
    catalog = make_star_catalog(stars={"star_1": star})
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")
    vis = make_spectroscopy_channel(channel_name="VIS")
    nir = make_photometry_channel(channel_name="NIR")

    def _fake_counts(photons_star, wavelengths, channel, ctx, bg_star):
        if channel.channel_name == "NIR":
            return np.array([4.0, 5.0], dtype=float)
        return np.array([1.0, 2.0, 3.0], dtype=float)

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range", return_value=(np.array([10.0, 20.0]), np.array([1000.0, 1100.0]))), \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel", side_effect=_fake_counts):
        result = populate_background_star_counts(catalog, nuv=nuv, vis=vis, nir=nir, ctx=ctx)

    assert ("star_1", "NUV") in result.counts_by_id_and_band
    assert ("star_1", "VIS") in result.counts_by_id_and_band
    assert ("star_1", "NIR") in result.counts_by_id_and_band
    assert isinstance(result.counts_by_id_and_band[("star_1", "NUV")], np.ndarray)
    assert isinstance(result.counts_by_id_and_band[("star_1", "VIS")], np.ndarray)
    assert isinstance(result.counts_by_id_and_band[("star_1", "NIR")], float)

    expected_spec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert result.counts_by_id_and_band[("star_1", "NUV")].dtype == np.float32
    np.testing.assert_allclose(result.counts_by_id_and_band[("star_1", "NUV")], expected_spec)
    assert result.counts_by_id_and_band[("star_1", "VIS")].dtype == np.float32
    np.testing.assert_allclose(result.counts_by_id_and_band[("star_1", "VIS")], expected_spec)

    expected_phot_sum = float(np.sum(np.array([4.0, 5.0], dtype=float)))
    assert result.counts_by_id_and_band[("star_1", "NIR")] == expected_phot_sum


# Tests: populate_background_star_counts
# Behavior: skips existing cached counts and logs the skip
def test_populate_background_star_counts_skips_existing_counts(make_star_catalog, make_star, make_spectroscopy_channel, make_photometry_channel, make_run_context):
    star = make_star(gaia_magnitude=12.0)
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")
    nir = make_photometry_channel(channel_name="NIR")
    existing = np.array([42.0], dtype=np.float32)
    catalog = make_star_catalog(stars={"star_1": star}, counts={("star_1", nuv.channel_name): existing})

    def _fake_counts(photons_star, wavelengths, channel, ctx, bg_star):
        assert channel.channel_name == "NIR"
        return np.array([1.0, 2.0, 3.0], dtype=float)

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range", return_value=(np.array([10.0, 20.0]), np.array([1000.0, 1100.0]))), \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel", side_effect=_fake_counts) as compute_mock:
        result = populate_background_star_counts(catalog, nuv=nuv, vis=None, nir=nir, ctx=ctx)

    assert result.counts_by_id_and_band[("star_1", "NUV")] is existing
    assert result.counts_by_id_and_band[("star_1", "NIR")] == 6.0
    assert compute_mock.call_count == 1
    called_channel = compute_mock.call_args_list[0].args[2]
    assert called_channel.channel_name == "NIR"


# Tests: populate_background_star_counts
# Behavior: computes counts for each background star in the catalog
def test_populate_background_star_counts_processes_multiple_stars(make_star_catalog, make_star, make_spectroscopy_channel, make_run_context):
    star_1 = make_star(name="HD 1", gaia_magnitude=12.0)
    star_2 = make_star(name="HD 2", gaia_magnitude=13.0)
    catalog = make_star_catalog(stars={"star_1": star_1, "star_2": star_2})
    ctx = make_run_context()
    nuv = make_spectroscopy_channel(channel_name="NUV")

    with patch("instrument.background_star_counts.get_required_wavelength_range", return_value=(1000.0, 2000.0)), \
         patch("instrument.background_star_counts.prepare_star_photon_flux_in_range", return_value=(np.array([10.0, 20.0]), np.array([1000.0, 1100.0]))), \
         patch("instrument.background_star_counts.compute_counts_per_s_px_one_channel", return_value=np.array([1.0, 2.0, 3.0], dtype=float)):
        result = populate_background_star_counts(catalog, nuv=nuv, vis=None, nir=None, ctx=ctx)

    assert ("star_1", nuv.channel_name) in result.counts_by_id_and_band
    assert ("star_2", nuv.channel_name) in result.counts_by_id_and_band