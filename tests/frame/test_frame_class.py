import numpy as np
import pytest
from astropy.io import fits
from unittest.mock import MagicMock, patch

from frame.frame_class import Frame


def test_frame_initialization_and_shape_properties():
    data = np.ones((4, 6))
    header = fits.Header()

    frame = Frame(data=data, header=header, frame_type="bias", channel_tag="NUV")

    assert frame.nx == 6
    assert frame.ny == 4
    assert frame.frame_type == "bias"
    assert frame.channel_tag == "NUV"


def test_frame_rejects_non_ndarray_data():
    header = fits.Header()
    with pytest.raises(ValueError):
        Frame(data=[[1, 2], [3, 4]], header=header, frame_type="bias", channel_tag="NUV")


def test_frame_rejects_non_2d_array():
    header = fits.Header()
    with pytest.raises(ValueError):
        Frame(data=np.ones(5), header=header, frame_type="bias", channel_tag="NUV")


def test_frame_rejects_non_fits_header_when_header_not_none():
    data = np.ones((2, 2))
    with pytest.raises(TypeError):
        Frame(data=data, header={"EXPTIME": 10.0}, frame_type="bias", channel_tag="NUV")


def test_write_fits_raises_when_header_is_none():
    data = np.ones((2, 2))
    frame = Frame(data=data, header=None, frame_type="bias", channel_tag="NUV")
    ctx = MagicMock()

    with pytest.raises(RuntimeError):
        frame.write_fits(ctx)


def test_write_fits_delegates_to_write_fits_frame():
    data = np.ones((2, 3))
    header = fits.Header()
    frame = Frame(data=data, header=header, frame_type="science", channel_tag="VIS")
    ctx = MagicMock()

    with patch("frame.write_fits.write_fits_frame") as mock_write:
        frame.write_fits(ctx)

    mock_write.assert_called_once_with(
        data,
        header,
        frame_type="science",
        channel_tag="VIS",
        ctx=ctx,
        index=0,
    )


def test_write_fits_passes_index():
    data = np.ones((2, 2))
    header = fits.Header()
    frame = Frame(data=data, header=header, frame_type="bias", channel_tag="NUV")
    ctx = MagicMock()

    with patch("frame.write_fits.write_fits_frame") as mock_write:
        frame.write_fits(ctx, index=3)

    mock_write.assert_called_once_with(
        data,
        header,
        frame_type="bias",
        channel_tag="NUV",
        ctx=ctx,
        index=3,
    )


def test_write_png_delegates_to_write_frame_png():
    data = np.ones((3, 3))
    header = fits.Header()
    frame = Frame(data=data, header=header, frame_type="dark", channel_tag="NUV")
    ctx = MagicMock()
    star = MagicMock()

    with patch("utils.images.write_frame_png") as mock_png:
        frame.write_png(ctx=ctx, star=star, show_stats=False)

    mock_png.assert_called_once_with(
        data,
        header,
        frame_type="dark",
        channel_tag="NUV",
        ctx=ctx,
        star=star,
        show_stats=False,
        index=0,
    )


def test_write_png_passes_index():
    data = np.ones((2, 2))
    header = fits.Header()
    frame = Frame(data=data, header=header, frame_type="science", channel_tag="VIS")
    ctx = MagicMock()
    star = MagicMock()

    with patch("utils.images.write_frame_png") as mock_png:
        frame.write_png(ctx=ctx, star=star, index=2)

    mock_png.assert_called_once()
    _, kwargs = mock_png.call_args
    assert kwargs["index"] == 2

