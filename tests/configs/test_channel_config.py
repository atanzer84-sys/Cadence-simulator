import pytest
from dataclasses import FrozenInstanceError

# verifies that a SpectroscopyChannel can be created via the fixture
# checks that selected overrides are applied correctly
# implicitly ensures all required fields are valid because construction succeeds
def test_spectroscopy_channel_init(make_spectroscopy_channel):
    ch = make_spectroscopy_channel(
        channel_name="NUV",
        x_pixels=2048,
        y_pixels=512,
        exposure_s=10.0,
    )

    assert ch.channel_name == "NUV"
    assert ch.exposure_s == 10.0
    assert ch.x_pixels == 2048

# verifies that a PhotometryChannel can be created via the fixture
# checks that PSF-related fields accept explicit overrides including None
# ensures basic attributes are stored correctly
def test_photometry_channel_init(make_photometry_channel):
    ch = make_photometry_channel(
        channel_name="NIR",
        x_pixels=100,
        y_pixels=100,
        exposure_s=5.0,
        psf_file="nir_psf.txt",
        psf_image=None,
        psf_center_x=None,
        psf_center_y=None,
        source_position_x_arcsec=None,
        source_position_y_arcsec=None,
    )

    assert ch.channel_name == "NIR"
    assert ch.exposure_s == 5.0
    assert ch.psf_file == "nir_psf.txt"
    assert ch.psf_image is None
    assert ch.psf_center_x is None
    assert ch.psf_center_y is None
    assert ch.source_position_x_arcsec is None
    assert ch.source_position_y_arcsec is None

# verifies that PhotometryChannel is immutable (frozen dataclass)
# attempts to modify a field after creation and expects a FrozenInstanceError
def test_photometry_channel_is_frozen(make_photometry_channel):
    ch = make_photometry_channel(
        channel_name="NIR",
        x_pixels=10,
        y_pixels=10,
        exposure_s=5.0,
        psf_file="nir_psf.txt",
    )

    with pytest.raises(FrozenInstanceError):
        ch.psf_file = "other.txt"