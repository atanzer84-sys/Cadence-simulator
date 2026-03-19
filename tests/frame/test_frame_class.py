import numpy as np
import pytest
from astropy.io import fits

from frame.frame_class import Frame


# Tests: Frame
# Behavior: valid construction stores data/header and exposes correct nx/ny properties
def test_frame_valid_creation_and_shape_properties():
    data = np.ones((2, 3))
    header = fits.Header()

    frame = Frame(data=data, header=header, frame_type="bias", channel_tag="NUV")

    assert frame.data is data
    assert frame.header is header
    assert frame.nx == 3
    assert frame.ny == 2


# Tests: Frame
# Behavior: data must be a numpy array, not a Python list
def test_frame_rejects_non_numpy_array():
    with pytest.raises(ValueError):
        Frame(data=[[1, 2], [3, 4]], header=None, frame_type="bias", channel_tag="NUV")


# Tests: Frame
# Behavior: data must be 2D; 1D and 3D arrays are rejected
def test_frame_rejects_non_2d_array():
    one_d = np.array([1, 2, 3])
    three_d = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        Frame(data=one_d, header=None, frame_type="bias", channel_tag="NUV")

    with pytest.raises(ValueError):
        Frame(data=three_d, header=None, frame_type="bias", channel_tag="NUV")


# Tests: Frame
# Behavior: header must be astropy.io.fits.Header or None
def test_frame_rejects_non_fits_header():
    data = np.ones((2, 2))

    with pytest.raises(TypeError):
        Frame(data=data, header="not a header", frame_type="bias", channel_tag="NUV")


# Tests: __post_init__, nx, ny
# Behavior: frame_type and channel_tag are stored and nx, ny reflect array shape
def test_frame_stores_frame_type_and_channel_tag():
    data = np.ones((7, 11))
    header = fits.Header()
    frame = Frame(data=data, header=header, frame_type="flat", channel_tag="IR")

    assert frame.frame_type == "flat"
    assert frame.channel_tag == "IR"
    assert frame.ny == 7
    assert frame.nx == 11


