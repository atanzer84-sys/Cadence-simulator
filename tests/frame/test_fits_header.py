import pytest
from datetime import datetime
from astropy.io import fits
from astropy.time import Time

from domain.star import Star
from frame.fits_header import initialize_fits_header, FITS_HEADER_KEYS


@pytest.fixture
def star_fixture():
    """Minimal Star via Star.from_params(); uses 'distance' (not distance_pc) per from_params API."""
    params = {
        "name": "TestStar",
        "right_ascension": 123.45,
        "declination": -54.321,
        "distance": 10.0,
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


def test_initialize_fits_header_returns_fits_header(star_fixture, fixed_timestamp):
    """initialize_fits_header returns a valid astropy fits.Header."""
    header = initialize_fits_header(star_fixture, fixed_timestamp)
    assert isinstance(header, fits.Header)


def test_header_contains_required_keys(star_fixture, fixed_timestamp):
    """All FITS header keys from fits_header.FITS_HEADER_KEYS are present (refactor-safe: single source of truth)."""
    header = initialize_fits_header(star_fixture, fixed_timestamp)

    for key in FITS_HEADER_KEYS:
        assert key in header, f"Missing header key: {key}"


def test_header_values_are_reasonable(star_fixture, fixed_timestamp):
    """Header values match the Star object and fixed metadata."""
    header = initialize_fits_header(star_fixture, fixed_timestamp)

    assert header["TELESCOP"] == "WALTzER"
    assert header["ROOTNAME"] == "WALTzER_output"
    assert header["TRGET"] == star_fixture.name
    assert header["TARGT_D"] == star_fixture.distance_pc
    assert header["VMAG"] == star_fixture.v_magnitude
    assert header["RA"] == pytest.approx(star_fixture.right_ascension)
    assert header["DEC"] == pytest.approx(star_fixture.declination)
    assert header["CCDTEMP"] == -50.0


def test_header_time_fields_are_valid_iso(star_fixture, fixed_timestamp):
    """Timestamp fields (EXP_STRT, DATEOBS, TIMEOBS) are valid ISO‑8601 and parseable by astropy."""
    header = initialize_fits_header(star_fixture, fixed_timestamp)

    exp_strt = header["EXP_STRT"]
    parsed = Time(exp_strt, format="isot")
    assert isinstance(parsed, Time)

    Time(f"{header['DATEOBS']}T{header['TIMEOBS']}", format="isot")
