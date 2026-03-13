from astroquery.simbad import Simbad


def find_stars(teff_min, teff_max, vmag_min, vmag_max, top=50):
    query = f"""
    SELECT TOP {top}
        b.main_id AS main_id,
        f.flux AS vmag,
        m.teff AS teff
    FROM basic AS b
    JOIN flux AS f
        ON f.oidref = b.oid
    JOIN mesFe_h AS m
        ON m.oidref = b.oid
    WHERE f.filter = 'V'
      AND m.teff >= {teff_min}
      AND m.teff <= {teff_max}
      AND f.flux >= {vmag_min}
      AND f.flux <= {vmag_max}
    ORDER BY vmag ASC, teff ASC
    """
    return Simbad.query_tap(query)


tbl = find_stars(3400, 3600, 10.5, 11.5, top=50)

tbl.pprint(max_lines=-1)