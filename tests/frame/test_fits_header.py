import pytest
from datetime import datetime
from astropy.io import fits
from astropy.time import Time
from frame.fits_header import initialize_fits_header
from domain.star import Star
import loaders.run_setup


@pytest.fixture
def star_fixture():
    """
    Creates a minimal, stable Star object using Star.from_params().
    """
    params = {
        "name": "TestStar",
        "right_ascension": 123.45,
        "declination": -54.321,
        "distance_pc": 10.0,
        "mass_sun_kg": 1.0,
        "v_magnitude": 5.0,
        "spectral_type": "G2V",
        "effective_temperature": 5800,
        "radius": 1.0,
        "mass": 1.0,
        "metallicity": 0.0,
        "surface_gravity": 4.4,
        "gaia_magnitude": 10.0,
        "log_r": 0.0,
        "radius_sun_cm": 69570000000.0,
    }
    required_keys = list(params.keys())
    return Star.from_params(params, required_keys=required_keys)


@pytest.fixture
def fixed_timestamp():
    """A deterministic timestamp for testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


def test_initialize_fits_header_returns_fits_header(monkeypatch, star_fixture, fixed_timestamp):
    # This test ensures initialize_fits_header returns a valid FITS header.
    monkeypatch.setattr(loaders.run_setup, "GLOBAL_TIMESTAMP", fixed_timestamp)

    header = initialize_fits_header(star_fixture)
    assert isinstance(header, fits.Header)


def test_header_contains_required_keys(monkeypatch, star_fixture, fixed_timestamp):
    # This test verifies that all mandatory FITS header keys are present.
    monkeypatch.setattr(loaders.run_setup, "GLOBAL_TIMESTAMP", fixed_timestamp)

    header = initialize_fits_header(star_fixture)

    required_keys = [
        "TELESCOP", "ROOTNAME", "EXP_STRT", "PRGRM_ID",
        "DATEOBS", "TIMEOBS", "JD", "MJD",
        "TRGET", "TARGT_ID", "TARGT_D", "TARGT_MS",
        "VMAG", "RA", "DEC", "GLAT", "GLON",
        "RA_HEX", "DEC_HEX", "CCDTEMP",
    ]

    for key in required_keys:
        assert key in header


def test_header_values_are_reasonable(monkeypatch, star_fixture, fixed_timestamp):
    # This test checks that header values match the Star object and fixed metadata.
    monkeypatch.setattr(loaders.run_setup, "GLOBAL_TIMESTAMP", fixed_timestamp)

    header = initialize_fits_header(star_fixture)

    assert header["TELESCOP"] == "WALTzER"
    assert header["ROOTNAME"] == "WALTzER_output"
    assert header["TRGET"] == star_fixture.name
    assert header["TARGT_D"] == star_fixture.distance_pc
    assert header["VMAG"] == star_fixture.v_magnitude
    assert header["RA"] == pytest.approx(star_fixture.right_ascension)
    assert header["DEC"] == pytest.approx(star_fixture.declination)
    assert header["CCDTEMP"] == -50.0


def test_header_time_fields_are_valid_iso(monkeypatch, star_fixture, fixed_timestamp):
    # This test ensures timestamp fields are valid ISO‑8601 and parseable by astropy.
    monkeypatch.setattr(loaders.run_setup, "GLOBAL_TIMESTAMP", fixed_timestamp)

    header = initialize_fits_header(star_fixture)

    exp_strt = header["EXP_STRT"]
    parsed = Time(exp_strt, format="isot")
    assert isinstance(parsed, Time)

    Time(f"{header['DATEOBS']}T{header['TIMEOBS']}", format="isot")
