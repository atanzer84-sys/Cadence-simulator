import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from frame.fits_header import (
    initialize_fits_header,
    append_image_stats_header,
    append_channel_frame_header,
    append_base_frame_header,
    append_photometry_header,
)

TEST_BASE_KEY = "BASEKEY"
TEST_BASE_VALUE = "base"
TEST_FILETYPE = "SCIENCE"
HDR_TELESCOPE = "TELESCOP"
HDR_ROOTNAME = "ROOTNAME"
HDR_EXP_START = "EXP_STRT"
HDR_PROGRAM_ID = "PRGRM_ID"
HDR_DATEOBS = "DATEOBS"
HDR_TIMEOBS = "TIMEOBS"
HDR_JD = "JD"
HDR_MJD = "MJD"
HDR_TRGET = "TRGET"
HDR_TARGT_ID = "TARGT_ID"
HDR_TARGT_D = "TARGT_D"
HDR_TARGT_MS = "TARGT_MS"
HDR_VMAG = "VMAG"
HDR_RA = "RA"
HDR_DEC = "DEC"
HDR_GLAT = "GLAT"
HDR_GLON = "GLON"
HDR_RA_HEX = "RA_HEX"
HDR_DEC_HEX = "DEC_HEX"
HDR_GEO_LAT = "GEO_LAT"
HDR_GEO_LON = "GEO_LON"
HDR_CCDTEMP = "CCDTEMP"

HDR_MEAN = "MEAN"
HDR_MEDIAN = "MEDIAN"
HDR_STDDEV = "STDDEV"
HDR_MAX = "MAX"
HDR_MIN = "MIN"

HDR_EXPTIME = "EXPTIME"
HDR_YCUT1 = "YCUT1"
HDR_YCUT2 = "YCUT2"
HDR_CCDGAIN = "CCDGAIN"
HDR_BIAS = "B_OFFSET"
HDR_RNOISE = "RNOISE"
HDR_DARKSIG = "DARKSIG"
HDR_DARKVAL = "DARKVAL"

HDR_FILETYPE = "FILETYPE"
HDR_CHANNEL = "CHANNEL"
HDR_EXP_ID = "EXP_ID"
HDR_OBS_ID = "OBS_ID"

HDR_CSTAR = "CSTAR"
HDR_CSTNOISE = "CSTNOISE"
HDR_PHOTX0 = "PHOTX0"
HDR_PHOTY0 = "PHOTY0"
HDR_PHOTINR = "PHOTINR"
HDR_PHOTOUTR = "PHOTOUTR"


def _count_key(header: fits.Header, key: str) -> int:
    return sum(1 for current_key in header.keys() if current_key == key)


EXPECTED_INITIALIZE_KEYS = {
    HDR_TELESCOPE,
    HDR_ROOTNAME,
    HDR_EXP_START,
    HDR_PROGRAM_ID,
    HDR_DATEOBS,
    HDR_TIMEOBS,
    HDR_JD,
    HDR_MJD,
    HDR_TRGET,
    HDR_TARGT_ID,
    HDR_TARGT_D,
    HDR_TARGT_MS,
    HDR_VMAG,
    HDR_RA,
    HDR_DEC,
    HDR_GLAT,
    HDR_GLON,
    HDR_RA_HEX,
    HDR_DEC_HEX,
    HDR_GEO_LAT,
    HDR_GEO_LON,
    HDR_CCDTEMP,
}


CONSTANT_METADATA_KEYS = {
    HDR_TELESCOPE,
    HDR_ROOTNAME,
    HDR_PROGRAM_ID,
    HDR_CCDTEMP,
    HDR_GEO_LAT,
    HDR_GEO_LON,
}


@pytest.fixture
def header_test_star(make_star):
    return make_star(
        name="TestStar",
        right_ascension=123.456789,
        declination=-54.321987,
        distance_pc=10.0,
        effective_temperature=5800,
        v_magnitude=5.0,
    )

# Tests: initialize_fits_header
# Behavior: returns a FITS header instance
def test_initialize_fits_header_returns_fits_header(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert isinstance(header, fits.Header)


# Tests: initialize_fits_header
# Behavior: rejects non-datetime timestamps
def test_initialize_fits_header_rejects_non_datetime_timestamp(header_test_star):
    with pytest.raises(TypeError):
        initialize_fits_header(header_test_star, "2024-01-01T12:00:00")


# Tests: initialize_fits_header
# Behavior: creates exactly the expected header contract keys
def test_initialize_fits_header_exact_key_contract(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert set(header.keys()) == EXPECTED_INITIALIZE_KEYS


# Tests: initialize_fits_header
# Behavior: constant metadata keys are present
def test_initialize_fits_header_constant_metadata_keys_present(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    for key in CONSTANT_METADATA_KEYS:
        assert key in header


# Tests: initialize_fits_header
# Behavior: target fields are copied from the star object
def test_initialize_fits_header_propagates_star_values(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert header[HDR_TRGET] == header_test_star.name
    assert header[HDR_TARGT_ID] == header_test_star.name
    assert header[HDR_TARGT_D] == header_test_star.distance_pc
    assert header[HDR_TARGT_MS] == header_test_star.effective_temperature
    assert header[HDR_VMAG] == header_test_star.v_magnitude


# Tests: initialize_fits_header
# Behavior: right ascension and declination are rounded to 4 decimal places
def test_initialize_fits_header_rounds_ra_dec(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert header[HDR_RA] == round(header_test_star.right_ascension, 4)
    assert header[HDR_DEC] == round(header_test_star.declination, 4)


# Tests: initialize_fits_header
# Behavior: JD and MJD are consistent with the provided timestamp
def test_initialize_fits_header_sets_jd_and_mjd(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)
    expected_time = Time(fixed_timestamp, scale="utc")

    assert header[HDR_JD] == pytest.approx(expected_time.jd)
    assert header[HDR_MJD] == pytest.approx(expected_time.mjd)


# Tests: initialize_fits_header
# Behavior: date and time header values are consistent with the provided timestamp
def test_initialize_fits_header_sets_date_and_time_fields(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert header[HDR_DATEOBS] == fixed_timestamp.date().isoformat()
    assert header[HDR_TIMEOBS] == fixed_timestamp.time().strftime("%H:%M:%S")

    parsed_exp_start = Time(header[HDR_EXP_START], format="isot", scale="utc")
    expected_time = Time(fixed_timestamp, scale="utc")

    assert parsed_exp_start.jd == pytest.approx(expected_time.jd)


# Tests: initialize_fits_header
# Behavior: hex coordinate fields are present as formatted strings
def test_initialize_fits_header_sets_hex_coordinate_fields(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert isinstance(header[HDR_RA_HEX], str)
    assert isinstance(header[HDR_DEC_HEX], str)
    assert ":" in header[HDR_RA_HEX]
    assert ":" in header[HDR_DEC_HEX]


# Tests: initialize_fits_header
# Behavior: galactic coordinate fields are present and numeric
def test_initialize_fits_header_sets_galactic_coordinate_fields(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert HDR_GLAT in header
    assert HDR_GLON in header
    assert isinstance(header[HDR_GLAT], float | int)
    assert isinstance(header[HDR_GLON], float | int)


# Tests: initialize_fits_header
# Behavior: geographic coordinate fields are present and numeric
def test_initialize_fits_header_sets_geographic_coordinate_fields(header_test_star, fixed_timestamp):
    header = initialize_fits_header(header_test_star, fixed_timestamp)

    assert HDR_GEO_LAT in header
    assert HDR_GEO_LON in header
    assert isinstance(header[HDR_GEO_LAT], float | int)
    assert isinstance(header[HDR_GEO_LON], float | int)



# Tests: append_image_stats_header
# Behavior: appends rounded image statistics to an existing header
def test_append_image_stats_header_appends_expected_stats():
    header = fits.Header()
    image = np.array([[1.111, 2.222], [3.333, 4.444]], dtype=float)

    append_image_stats_header(header, image)

    assert header[HDR_MEAN] == round(float(np.mean(image)), 2)
    assert header[HDR_MEDIAN] == round(float(np.median(image)), 2)
    assert header[HDR_STDDEV] == round(float(np.std(image)), 2)
    assert header[HDR_MAX] == round(float(np.max(image)), 2)
    assert header[HDR_MIN] == round(float(np.min(image)), 2)


# Tests: append_image_stats_header
# Behavior: rejects non-header types when header is provided
def test_append_image_stats_header_rejects_non_fits_header():
    with pytest.raises(TypeError):
        append_image_stats_header("not-a-header", np.ones((2, 2), dtype=float))


# Tests: append_image_stats_header
# Behavior: rejects image values that are not numpy arrays
def test_append_image_stats_header_rejects_non_ndarray_image():
    with pytest.raises(TypeError):
        append_image_stats_header(fits.Header(), [[1.0, 2.0], [3.0, 4.0]])


# Tests: append_image_stats_header
# Behavior: rejects non-2D arrays
def test_append_image_stats_header_rejects_non_2d_image():
    with pytest.raises(ValueError):
        append_image_stats_header(fits.Header(), np.ones(4, dtype=float))


# Tests: append_image_stats_header
# Behavior: rejects non-numeric array dtypes
def test_append_image_stats_header_rejects_non_numeric_image_dtype():
    with pytest.raises(TypeError):
        append_image_stats_header(fits.Header(), np.array([["a", "b"], ["c", "d"]], dtype=object))


# Tests: append_image_stats_header
# Behavior: replaces existing statistic cards instead of creating duplicates
def test_append_image_stats_header_replaces_existing_cards_without_duplicates():
    header = fits.Header()
    image = np.array([[1.111, 2.222], [3.333, 4.444]], dtype=float)

    append_image_stats_header(header, image)
    append_image_stats_header(header, image)

    assert _count_key(header, HDR_MEAN) == 1
    assert _count_key(header, HDR_MEDIAN) == 1
    assert _count_key(header, HDR_STDDEV) == 1
    assert _count_key(header, HDR_MAX) == 1
    assert _count_key(header, HDR_MIN) == 1


# Tests: append_image_stats_header
# Behavior: does nothing when header is None
def test_append_image_stats_header_with_none_header_does_nothing():
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    append_image_stats_header(None, image)


# Tests: append_channel_frame_header
# Behavior: appends common frame metadata including dark values by default
def test_append_channel_frame_header_appends_expected_fields(make_spectroscopy_channel):
    header = fits.Header()
    channel = make_spectroscopy_channel(
        y_pixels=32,
        ccd_gain=1.5,
        bias_offset=200.0,
        read_noise=3.0,
        dark_current_sigma=0.02,
        dark_noise=1.0,
    )
    exptime_s = 10.0

    append_channel_frame_header(header, channel, exptime_s=exptime_s)

    assert header[HDR_EXPTIME] == exptime_s
    assert header[HDR_YCUT1] == 0
    assert header[HDR_YCUT2] == channel.y_pixels - 1
    assert header[HDR_CCDGAIN] == channel.ccd_gain
    assert header[HDR_BIAS] == channel.bias_offset
    assert header[HDR_RNOISE] == channel.read_noise
    assert header[HDR_DARKSIG] == channel.dark_current_sigma
    assert header[HDR_DARKVAL] == channel.dark_noise


# Tests: append_channel_frame_header
# Behavior: replaces existing frame metadata cards instead of creating duplicates
def test_append_channel_frame_header_replaces_existing_cards_without_duplicates(make_spectroscopy_channel):
    header = fits.Header()
    channel = make_spectroscopy_channel()

    append_channel_frame_header(header, channel, exptime_s=10.0)
    append_channel_frame_header(header, channel, exptime_s=20.0)

    assert _count_key(header, HDR_EXPTIME) == 1
    assert _count_key(header, HDR_YCUT1) == 1
    assert _count_key(header, HDR_YCUT2) == 1
    assert _count_key(header, HDR_CCDGAIN) == 1
    assert _count_key(header, HDR_BIAS) == 1
    assert _count_key(header, HDR_RNOISE) == 1
    assert _count_key(header, HDR_DARKSIG) == 1
    assert _count_key(header, HDR_DARKVAL) == 1
    assert header[HDR_EXPTIME] == 20.0


# Tests: append_channel_frame_header
# Behavior: omits dark fields when include_dark is False
def test_append_channel_frame_header_omits_dark_fields_when_disabled(make_spectroscopy_channel):
    header = fits.Header()
    channel = make_spectroscopy_channel()
    exptime_s = 10.0

    append_channel_frame_header(header, channel, exptime_s=exptime_s, include_dark=False)

    assert header[HDR_EXPTIME] == exptime_s
    assert header[HDR_YCUT1] == 0
    assert header[HDR_YCUT2] == channel.y_pixels - 1
    assert header[HDR_CCDGAIN] == channel.ccd_gain
    assert header[HDR_BIAS] == channel.bias_offset
    assert header[HDR_RNOISE] == channel.read_noise
    assert HDR_DARKSIG not in header
    assert HDR_DARKVAL not in header


# Tests: append_channel_frame_header
# Behavior: does nothing when header is None
def test_append_channel_frame_header_with_none_header_does_nothing(make_spectroscopy_channel):
    channel = make_spectroscopy_channel()

    append_channel_frame_header(None, channel, exptime_s=10.0)


# Tests: append_base_frame_header
# Behavior: returns a copied header with identity fields and 1 based numbering
def test_append_base_frame_header_returns_copied_header_with_identity_fields(make_spectroscopy_channel):
    base_header = fits.Header()
    base_header[TEST_BASE_KEY] = TEST_BASE_VALUE
    channel = make_spectroscopy_channel()
    filetype = TEST_FILETYPE
    index0 = 0

    header = append_base_frame_header(base_header, filetype, channel, index0=index0)

    assert header is not base_header
    assert header[TEST_BASE_KEY] == TEST_BASE_VALUE
    assert header[HDR_FILETYPE] == filetype
    assert header[HDR_CHANNEL] == channel.channel_name
    assert header[HDR_EXP_ID] == f"{filetype} {index0 + 1}"
    assert header[HDR_OBS_ID] == f"Obs {filetype} {index0 + 1}"
    assert HDR_FILETYPE not in base_header
    assert HDR_CHANNEL not in base_header
    assert HDR_EXP_ID not in base_header
    assert HDR_OBS_ID not in base_header


# Tests: append_base_frame_header
# Behavior: replaces existing identity cards in the returned header copy
def test_append_base_frame_header_replaces_existing_cards_without_duplicates(make_spectroscopy_channel):
    base_header = fits.Header()
    base_header[HDR_FILETYPE] = "OLD"
    base_header[HDR_CHANNEL] = "OLD"
    base_header[HDR_EXP_ID] = "OLD"
    base_header[HDR_OBS_ID] = "OLD"
    channel = make_spectroscopy_channel(channel_name="NUV")

    header = append_base_frame_header(base_header, TEST_FILETYPE, channel, index0=1)

    assert _count_key(header, HDR_FILETYPE) == 1
    assert _count_key(header, HDR_CHANNEL) == 1
    assert _count_key(header, HDR_EXP_ID) == 1
    assert _count_key(header, HDR_OBS_ID) == 1
    assert header[HDR_FILETYPE] == TEST_FILETYPE
    assert header[HDR_CHANNEL] == channel.channel_name
    assert header[HDR_EXP_ID] == f"{TEST_FILETYPE} 2"
    assert header[HDR_OBS_ID] == f"Obs {TEST_FILETYPE} 2"


# Tests: append_base_frame_header
# Behavior: returns None when base_header is None
def test_append_base_frame_header_returns_none_when_base_header_is_none(make_spectroscopy_channel):
    channel = make_spectroscopy_channel()
    filetype = TEST_FILETYPE

    header = append_base_frame_header(None, filetype, channel, index0=0)

    assert header is None


# Tests: append_photometry_header
# Behavior: appends rounded photometry values and center coordinates
def test_append_photometry_header_appends_expected_fields():
    header = fits.Header()
    cstar = 1234.567
    cstar_noise = 12.3456
    phot_x0 = 17
    phot_y0 = 9
    phot_inr = 4.567
    phot_outr = 7.891
    phot = (cstar, cstar_noise, phot_x0, phot_y0, phot_inr, phot_outr)

    append_photometry_header(header, phot)

    assert header[HDR_CSTAR] == round(cstar, 2)
    assert header[HDR_CSTNOISE] == round(cstar_noise, 2)
    assert header[HDR_PHOTX0] == phot_x0
    assert header[HDR_PHOTY0] == phot_y0
    assert header[HDR_PHOTINR] == round(phot_inr, 2)
    assert header[HDR_PHOTOUTR] == round(phot_outr, 2)


# Tests: append_photometry_header
# Behavior: replaces existing photometry cards instead of creating duplicates
def test_append_photometry_header_replaces_existing_cards_without_duplicates():
    header = fits.Header()
    first_phot = (1.0, 2.0, 3, 4, 5.0, 6.0)
    second_phot = (11.111, 22.222, 7, 8, 9.999, 10.101)

    append_photometry_header(header, first_phot)
    append_photometry_header(header, second_phot)

    assert _count_key(header, HDR_CSTAR) == 1
    assert _count_key(header, HDR_CSTNOISE) == 1
    assert _count_key(header, HDR_PHOTX0) == 1
    assert _count_key(header, HDR_PHOTY0) == 1
    assert _count_key(header, HDR_PHOTINR) == 1
    assert _count_key(header, HDR_PHOTOUTR) == 1
    assert header[HDR_CSTAR] == round(second_phot[0], 2)
    assert header[HDR_CSTNOISE] == round(second_phot[1], 2)
    assert header[HDR_PHOTX0] == second_phot[2]
    assert header[HDR_PHOTY0] == second_phot[3]
    assert header[HDR_PHOTINR] == round(second_phot[4], 2)
    assert header[HDR_PHOTOUTR] == round(second_phot[5], 2)


# Tests: append_photometry_header
# Behavior: does nothing when header is None
def test_append_photometry_header_with_none_header_does_nothing():
    phot = (1234.567, 12.3456, 17, 9, 4.567, 7.891)

    append_photometry_header(None, phot)


# Tests: append_photometry_header
# Behavior: does nothing when photometry tuple is None
def test_append_photometry_header_with_none_phot_does_nothing():
    header = fits.Header()

    append_photometry_header(header, None)

    assert HDR_CSTAR not in header
    assert HDR_CSTNOISE not in header
    assert HDR_PHOTX0 not in header
    assert HDR_PHOTY0 not in header
    assert HDR_PHOTINR not in header
    assert HDR_PHOTOUTR not in header