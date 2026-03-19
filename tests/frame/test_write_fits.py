import numpy as np
from astropy.io import fits
from frame.write_fits import write_fits_frame
import pytest

TEST_FILETYPE = "science"
TEST_CHANNEL = "nuv"


@pytest.fixture
def make_static_frame(make_frame):
    def _make():
        return make_frame(
            data=np.ones((5, 5), dtype=float),
            header=fits.Header(),
            frame_type=TEST_FILETYPE,
            channel_tag=TEST_CHANNEL,
        )
    return _make


# Tests: write_fits_frame
# Behavior: writes a FITS file to the path returned by build_base_output_path
def test_write_fits_frame_writes_file(make_static_frame, make_run_context, tmp_path, monkeypatch):
    frame = make_static_frame()
    ctx = make_run_context(output_dir=tmp_path)
    fake_path = tmp_path / "test.fits"

    monkeypatch.setattr("frame.write_fits.build_base_output_path", lambda *a, **k: fake_path)

    write_fits_frame(frame, ctx, index=0, exposure=1.0)


    assert fake_path.exists()


# Tests: write_fits_frame
# Behavior: appends filename to header and mutates original header
def test_write_fits_frame_appends_filename(make_static_frame, make_run_context, tmp_path, monkeypatch):
    frame = make_static_frame()
    ctx = make_run_context(output_dir=tmp_path)
    fake_path = tmp_path / "test.fits"

    monkeypatch.setattr("frame.write_fits.build_base_output_path", lambda *a, **k: fake_path)

    write_fits_frame(frame, ctx, index=0, exposure=1.0)

    assert "FILENAME" in frame.header
    assert frame.header["FILENAME"] == fake_path.name


# Tests: write_fits_frame
# Behavior: preserves data and header content when written to disk
def test_write_fits_frame_preserves_data_and_header(make_static_frame, make_run_context, tmp_path, monkeypatch):
    frame = make_static_frame()
    frame.header["TESTKEY"] = 123

    ctx = make_run_context(output_dir=tmp_path)
    fake_path = tmp_path / "test.fits"

    monkeypatch.setattr("frame.write_fits.build_base_output_path", lambda *a, **k: fake_path)

    write_fits_frame(frame, ctx, index=0, exposure=1.0)

    with fits.open(fake_path) as hdul:
        np.testing.assert_allclose(hdul[0].data, frame.data)
        assert hdul[0].header["TESTKEY"] == 123


# Tests: write_fits_frame
# Behavior: overwrites existing file
def test_write_fits_frame_overwrites_existing(make_static_frame, make_run_context, tmp_path, monkeypatch):
    frame = make_static_frame()
    ctx = make_run_context(output_dir=tmp_path)
    fake_path = tmp_path / "test.fits"

    fake_path.write_bytes(b"old")

    monkeypatch.setattr("frame.write_fits.build_base_output_path", lambda *a, **k: fake_path)

    write_fits_frame(frame, ctx, index=0, exposure=1.0)

    assert fake_path.exists()
    assert fake_path.stat().st_size > 0


# Tests: write_fits_frame
# Behavior: logs write operation at INFO level
def test_write_fits_frame_logs_info(make_static_frame, make_run_context, tmp_path, monkeypatch, caplog):
    frame = make_static_frame()
    ctx = make_run_context(output_dir=tmp_path)
    fake_path = tmp_path / "test.fits"

    monkeypatch.setattr("frame.write_fits.build_base_output_path", lambda *a, **k: fake_path)

    caplog.set_level("INFO")

    write_fits_frame(frame, ctx, index=0, exposure=1.0)

    assert any(r.levelname == "INFO" for r in caplog.records)