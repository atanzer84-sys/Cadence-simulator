import numpy as np
import pytest

from loaders import effective_area_loader


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_ok_parses_pixel_scale_and_first_last_columns(monkeypatch, tmp_path):
    # Verifies that a valid file returns wavelength from the first column, effective area from the last column, and the parsed pixel scale.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_ok.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# header",
                "# Pixel scale: 1.23e-2",
                "Wavelength col2 col3 EffectiveArea",
                "1000  1  2  0.10",
                "1100  3  4  0.20",
                "1200  5  6  0.30",
            ]
        )
        + "\n",
    )

    wl, ea, pixel_scale = effective_area_loader.load_effective_area_file(fname)

    assert np.allclose(wl, np.array([1000.0, 1100.0, 1200.0]))
    assert np.allclose(ea, np.array([0.10, 0.20, 0.30]))
    assert pixel_scale == pytest.approx(1.23e-2)
    assert len(wl) == len(ea)


def test_missing_file_raises_valueerror(monkeypatch, tmp_path):
    # Verifies that a non-existent effective area file raises ValueError with a clear message.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    (tmp_path / "data").mkdir()

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file("does_not_exist.txt")

    assert "Effective area file not found" in str(exc.value)


def test_missing_pixel_scale_header_raises_valueerror(monkeypatch, tmp_path):
    # Verifies that the loader fails if the required '# Pixel scale:' header line is missing.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_missing_pixelscale.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# header but no pixel scale",
                "Wavelength EffectiveArea",
                "1000  0.1",
                "1100  0.2",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Missing required header line" in str(exc.value)
    assert "Pixel scale" in str(exc.value)


def test_invalid_pixel_scale_value_raises_valueerror(monkeypatch, tmp_path):
    # Verifies that a non-numeric pixel scale value raises ValueError.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_bad_pixelscale.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: not_a_number",
                "Wavelength EffectiveArea",
                "1000  0.1",
                "1100  0.2",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Invalid pixel scale value" in str(exc.value)


def test_no_numeric_rows_raises_valueerror(monkeypatch, tmp_path):
    # Verifies that a file with no numeric data rows raises ValueError from the numeric-row search.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_no_numeric.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.01",
                "Wavelength EffectiveArea",
                "notnumeric stillnotnumeric",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Could not find any numeric data rows" in str(exc.value)


def test_one_column_table_raises_valueerror_invalid_structure(monkeypatch, tmp_path):
    # Verifies that a one-column numeric table raises ValueError via the data.ndim == 1 guard (prevents the (1,N) reshape bug).
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_one_column.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.417684",
                "Wavelength",
                "2397.0000",
                "2397.4177",
                "2397.8354",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Invalid effective area table structure" in str(exc.value)


def test_one_row_two_columns_raises_valueerror_invalid_structure(monkeypatch, tmp_path):
    # Verifies that a single numeric row (even with two columns) is rejected because np.loadtxt returns 1D and the loader requires a 2D table.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_one_row.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.5",
                "Wavelength EffectiveArea",
                "1500  0.42",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Invalid effective area table structure" in str(exc.value)


def test_header_lines_after_pixel_scale_do_not_break_parsing(monkeypatch, tmp_path):
    # Verifies that extra comment lines after the pixel scale header do not prevent numeric data parsing.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_extra_header.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.02",
                "# another comment after pixel scale",
                "# yet another comment",
                "Wavelength foo bar EffectiveArea",
                "1000  7  8  0.11",
                "1100  9 10  0.22",
            ]
        )
        + "\n",
    )

    wl, ea, pixel_scale = effective_area_loader.load_effective_area_file(fname)
    assert pixel_scale == pytest.approx(0.02)
    assert np.allclose(wl, np.array([1000.0, 1100.0]))
    assert np.allclose(ea, np.array([0.11, 0.22]))


def test_extra_numeric_columns_first_and_last_used(monkeypatch, tmp_path):
    # Verifies that with extra numeric columns the loader still uses first column as wavelength and last column as effective area.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_extra_cols.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.01",
                "Wavelength c2 c3 c4 EffectiveArea",
                "1000  1  2  3  0.50",
                "1100  4  5  6  0.60",
            ]
        )
        + "\n",
    )

    wl, ea, _ = effective_area_loader.load_effective_area_file(fname)
    assert np.allclose(wl, np.array([1000.0, 1100.0]))
    assert np.allclose(ea, np.array([0.50, 0.60]))


def test_leading_trailing_whitespace_ok(monkeypatch, tmp_path):
    # Verifies that whitespace and tabs around numeric values do not break parsing.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_whitespace.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.01",
                "Wavelength EffectiveArea",
                "   1000\t   0.10   ",
                "\t1100\t0.20\t",
            ]
        )
        + "\n",
    )

    wl, ea, _ = effective_area_loader.load_effective_area_file(fname)
    assert np.allclose(wl, np.array([1000.0, 1100.0]))
    assert np.allclose(ea, np.array([0.10, 0.20]))


def test_blank_lines_inside_numeric_block_ok(monkeypatch, tmp_path):
    # Verifies that blank lines between numeric rows are tolerated and do not truncate the table.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_blank_lines.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.01",
                "Wavelength EffectiveArea",
                "1000  0.10",
                "",
                "1100  0.20",
                "   ",
                "1200  0.30",
            ]
        )
        + "\n",
    )

    wl, ea, _ = effective_area_loader.load_effective_area_file(fname)
    assert np.allclose(wl, np.array([1000.0, 1100.0, 1200.0]))
    assert np.allclose(ea, np.array([0.10, 0.20, 0.30]))


def test_malformed_numeric_row_raises_valueerror(monkeypatch, tmp_path):
    # Verifies that a malformed numeric row causes parsing to fail rather than silently skipping data.
    monkeypatch.setattr(effective_area_loader, "get_repo_root", lambda base_dir=None: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fname = "ea_malformed_row.txt"
    _write(
        data_dir / fname,
        "\n".join(
            [
                "# Pixel scale: 0.01",
                "Wavelength foo EffectiveArea",
                "1000  1  0.10",
                "1100  2  BAD",
                "1200  3  0.30",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError) as exc:
        effective_area_loader.load_effective_area_file(fname)

    assert "Failed to parse numeric data" in str(exc.value)
