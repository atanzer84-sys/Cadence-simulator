from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from datetime import datetime
from loaders.run_cadence_context import setup_output_directory
from pathlib import Path
import pandas as pd


def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)


def resolve_target_center(target: str | int):
    Gaia.TIMEOUT = 120
    target_str = str(target).strip()

    if target_str.isdigit():
        ts("Resolving Gaia source_id...")
        query = f"""
        SELECT source_id, ra, dec, parallax, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE source_id = {target_str}
        """
        job = Gaia.launch_job_async(query)
        tbl = job.get_results()

        if len(tbl) == 0:
            raise ValueError(f"No Gaia source found for source_id={target_str}")

        ra0 = float(tbl["ra"][0])
        dec0 = float(tbl["dec"][0])
        coord = SkyCoord(ra=ra0, dec=dec0, unit="deg", frame="icrs")
        resolved_source_id = int(tbl["source_id"][0])

        ts("Target RA deg:", ra0)
        ts("Target DEC deg:", dec0)

        return coord, resolved_source_id

    ts("Resolving target name...")
    coord = SkyCoord.from_name(target_str)
    ts("Target RA deg:", coord.ra.deg)
    ts("Target DEC deg:", coord.dec.deg)

    return coord, None


def query_gaia_nearby_with_params(coord: SkyCoord, radius_arcsec: float, mag_limit: float):
    Gaia.TIMEOUT = 120

    ra0 = coord.ra.deg
    dec0 = coord.dec.deg
    radius_arcmin = radius_arcsec / 60.0
    radius_deg = radius_arcsec / 3600.0

    ts(f"Running cone search on gaia_source ({radius_arcsec} arcsec = {radius_arcmin:.2f} arcmin)...")

    query_cone = f"""
    SELECT source_id, ra, dec, parallax, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg})
    )
    AND phot_g_mean_mag < {mag_limit}
    """

    job1 = Gaia.launch_job_async(query_cone)
    cone = job1.get_results()

    if len(cone) == 0:
        ts("Cone search done. Rows: 0")
        return cone

    ts(f"Cone search done. Rows: {len(cone)}")

    ids_sql = ",".join(str(int(x)) for x in cone["source_id"])

    ts("Querying astrophysical_parameters for", len(cone), "source_ids...")

    query_ap = f"""
    SELECT
        gs.source_id,
        gs.ra,
        gs.dec,
        gs.parallax,
        gs.phot_g_mean_mag,
        COALESCE(ap.teff_gspphot, ap.teff_gspspec, supp.teff_gspspec_ann, gs.rv_template_teff) AS Teff,
        COALESCE(ap.distance_gspphot, supp.distance_gspphot_phoenix, supp.distance_gspphot_marcs) AS dist_pc
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
    LEFT JOIN gaiadr3.astrophysical_parameters_supp AS supp ON gs.source_id = supp.source_id
    WHERE gs.source_id IN ({ids_sql})
    """

    job2 = Gaia.launch_job_async(query_ap)
    out = job2.get_results()

    ts("Combined query returned rows:", len(out))

    out.sort("phot_g_mean_mag")
    return out


def gaia_nearby_stars_with_params(target: str | int, radius_arcsec: float, mag_limit: float):
    coord, resolved_source_id = resolve_target_center(target)
    out = query_gaia_nearby_with_params(coord, radius_arcsec, mag_limit)
    return out, coord, resolved_source_id


def mark_target_and_separation(tbl, coord: SkyCoord, resolved_source_id: int | None):
    if len(tbl) == 0:
        return tbl

    sources_coord = SkyCoord(ra=tbl["ra"], dec=tbl["dec"], unit="deg", frame="icrs")
    separations = coord.separation(sources_coord)
    tbl["sep_arcsec"] = separations.arcsec

    is_target = [False] * len(tbl)

    if resolved_source_id is not None:
        matches = [int(x) == int(resolved_source_id) for x in tbl["source_id"]]
        if any(matches):
            idx_target = matches.index(True)
        else:
            idx_target = separations.argmin()
    else:
        idx_target = separations.argmin()

    is_target[idx_target] = True
    tbl["is_target"] = is_target

    return tbl


def build_master_excel_from_csvs(output_dir):
    csv_files = sorted(Path(output_dir).glob("*.csv"))

    if not csv_files:
        print("No CSV files found for Excel export.")
        return

    all_dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        target_star = csv_file.stem.replace("_", " ")
        df.insert(0, "target_star", target_star)
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    if "sep_arcsec" in master_df.columns:
        master_df = master_df.sort_values(["target_star", "sep_arcsec"], ascending=[True, True])

    if "source_id" in master_df.columns:
        master_df["source_id"] = master_df["source_id"].astype(str)

    excel_path = Path(output_dir) / "all_background_stars.xlsx"
    master_df.to_excel(excel_path, index=False)

    print(f"Saved master Excel file to {excel_path}")


def main(existing_output_dir=None):
    targets = [
        "HD 2685",
        "KELT-9",
        "TIC 393818343",
        "WASP-189",
        "WASP-69",
        "HAT-P-1",
        "HAT-P-14",
        "HAT-P-2",
        "HAT-P-22",
        "HAT-P-60",
        "HAT-P-69",
        "HAT-P-70",
        "HD 118203",
        "HD 1397",
        "HD 149026",
        "HD 152843",
        "HD 189733",
        "HD 202772 A",
        "HD 209458",
        "HD 219666",
        "HD 221416",
        "HD 332231",
        "HD 88133",
        "HD 89345",
        "K2-232",
        "KELT-11",
        "KELT-17",
        "KELT-19 A",
        "KELT-2 A",
        "KELT-20",
        "KELT-24",
        "KELT-3",
        "KELT-4 A",
        "KELT-7",
        "KOI-13",
        "LTT 9779",
        "MASCARA-1",
        "MASCARA-4",
        "TOI-1135",
        "TOI-1136",
        "TOI-1333",
        "TOI-1408",
        "TOI-1431",
        "TOI-1518",
        "TOI-1789",
        "TOI-1842",
        "TOI-2005",
        "TOI-2145",
        "TOI-2497",
        "TOI-257",
        "TOI-421",
        "TOI-4551",
        "TOI-4603",
        "TOI-481",
        "TOI-5108",
        "TOI-5398",
        "TOI-6038 A",
        "TOI-622",
        "TOI-677",
        "TOI-778",
        "WASP-131",
        "WASP-136",
        "WASP-14",
        "WASP-166",
        "WASP-178",
        "WASP-18",
        "WASP-33",
        "WASP-38",
        "WASP-7",
        "WASP-74",
        "WASP-76",
        "WASP-79",
        "WASP-8",
        "WASP-82",
        "WASP-94 A",
        "WASP-95",
        "WASP-99",
        "XO-3",
        "WASP-12",
        "HAT-P-32",
        "Kepler-13",
        "TrES-4",
        "HD 189733",
        "HD 189733 B",
        "HAT-P-7",
        # "4093826354004454784",
    ]

    RADIUS_ARCSEC = 600.0
    MAG_LIMIT = 20.0

    if existing_output_dir is None:
        output_dir, _, _ = setup_output_directory()
        run_queries = True
    else:
        output_dir = Path(existing_output_dir)
        run_queries = False

    if run_queries:
        for target in targets:
            print(f"Processing {target}")

            try:
                tbl, coord, resolved_source_id = gaia_nearby_stars_with_params(
                    target=target,
                    radius_arcsec=RADIUS_ARCSEC,
                    mag_limit=MAG_LIMIT,
                )

                tbl = mark_target_and_separation(tbl, coord, resolved_source_id)

                filename = str(target).replace(" ", "_") + ".csv"
                filepath = output_dir / filename

                tbl.write(filepath, format="ascii.csv", overwrite=True)

                print(f"Saved {len(tbl)} rows to {filepath}")

            except Exception as e:
                print(f"Failed for {target}: {e}")

    build_master_excel_from_csvs(output_dir)


if __name__ == "__main__":
    main()
    # main("/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/Faint")