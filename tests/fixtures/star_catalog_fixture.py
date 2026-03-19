import pytest

from domain.star_catalog import StarCatalog


@pytest.fixture
def make_star_catalog():
    def _make_star_catalog(*, stars=None, offsets=None, counts=None):
        catalog = StarCatalog()

        if stars is not None:
            for star_id, star in stars.items():
                catalog.add_star(star_id, star)

        if offsets is not None:
            for star_id, (dx_arcsec, dy_arcsec) in offsets.items():
                catalog.set_offset_arcsec(star_id, dx_arcsec, dy_arcsec)

        if counts is not None:
            for key, value in counts.items():
                catalog.counts_by_id_and_band[key] = value

        return catalog

    return _make_star_catalog