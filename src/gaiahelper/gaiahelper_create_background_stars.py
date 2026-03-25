from astropy.coordinates import SkyCoord
from datetime import datetime
from loaders.load_gaia import _gaia_cone_search, _gaia_query_for_source_ids, _run_gaia_query
from loaders.run_waltzer_context import setup_output_directory
from pathlib import Path
import pandas as pd


OUTPUT_HEADER_MAP = {}


def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)


def rename_output_columns(tbl):
    if tbl is None:
        return None
    renamed = tbl.copy()
    for old_name, new_name in OUTPUT_HEADER_MAP.items():
        if old_name in renamed.colnames and new_name not in renamed.colnames:
            renamed.rename_column(old_name, new_name)
    return renamed


def gaia_nearby_stars_with_params_by_name(target_name: str, radius_arcsec: float, mag_limit: float, GAIA_USE_ASYNC_JOBS: bool = True):
    ts("Resolving target name...")

    coord = SkyCoord.from_name(target_name)
    ts("Target RA deg:", coord.ra.deg)
    ts("Target DEC deg:", coord.dec.deg)

    radius_arcmin = radius_arcsec / 60.0
    ts(f"Running cone search on gaia_source ({radius_arcsec} arcsec = {radius_arcmin:.2f} arcmin)...")

    cone = _gaia_cone_search(coord, radius_arcsec=radius_arcsec, g_mag_limit=mag_limit, GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS)

    if cone is None or len(cone) == 0:
        return cone, coord

    ts(f"Cone search done. Rows: {len(cone)}")

    ids = [int(x) for x in cone["source_id"]]
    ts("Querying astrophysical_parameters for", len(ids), "source_ids...")

    query = _gaia_query_for_source_ids(ids, g_mag_limit=mag_limit)
    out = _run_gaia_query(query, GAIA_USE_ASYNC_JOBS)

    if out is None:
        raise RuntimeError(f"Gaia joined query returned None for {target_name}")

    ts("Combined query returned rows:", len(out))

    out.sort("phot_g_mean_mag")
    return out, coord


def split_target_and_background(tbl, coord: SkyCoord):
    if len(tbl) == 0:
        return tbl, None

    sources_coord = SkyCoord(ra=tbl["ra"], dec=tbl["dec"], unit="deg", frame="icrs")
    separations = coord.separation(sources_coord)
    tbl["sep_arcsec"] = separations.arcsec

    idx_target = separations.argmin()

    target_row = tbl[idx_target:idx_target + 1].copy()
    mask = [i != idx_target for i in range(len(tbl))]
    background_tbl = tbl[mask].copy()

    return background_tbl, target_row


def write_targets_excel(target_rows, output_dir):
    if not target_rows:
        print("No target rows collected for Excel export.")
        return

    all_target_dfs = []

    for target_name, target_row in target_rows:
        renamed_target_row = rename_output_columns(target_row)
        df = renamed_target_row.to_pandas()
        df.insert(0, "target_star", target_name)

        if "source_id" in df.columns:
            df["source_id"] = df["source_id"].astype(str)

        all_target_dfs.append(df)

    target_df = pd.concat(all_target_dfs, ignore_index=True)
    for column in target_df.columns:
        if column in ("target_star", "source_id"):
            continue
        target_df[column] = pd.to_numeric(target_df[column], errors="coerce")

    excel_path = Path(output_dir) / "all_target_stars.xlsx"
    target_df.to_excel(excel_path, index=False)

    print(f"Saved target stars Excel file to {excel_path}")


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


def report_saved_vs_missing(star_names, output_dir):
    csv_files = sorted(Path(output_dir).glob("*.csv"))
    have_files = set()
    for csv_file in csv_files:
        target_star = csv_file.stem.replace("_", " ")
        have_files.add(target_star)

    expected = set(star_names)
    succeeded = sorted(expected & have_files)
    missing = sorted(expected - have_files)

    print("==== Gaia background CSV report ====")
    print(f"Total targets: {len(expected)}")
    print(f"With CSV: {len(succeeded)}")
    print(f"Without CSV: {len(missing)}")

    if succeeded:
        print("Targets with CSV:")
        for name in succeeded:
            print(" -", name)

    if missing:
        print("Targets without CSV (likely Gaia failures or empty cones):")
        for name in missing:
            print(" -", name)


def main(existing_output_dir=None):
    # star_names = [
    #     "HD 2685",
    #     "KELT-9",
    #     "TIC 393818343",
    #     "WASP-189",
    #     "WASP-69",
    #     "HAT-P-1",
    #     "HAT-P-14",
    #     "HAT-P-2",
    #     "HAT-P-22",
    #     "HAT-P-60",
    #     "HAT-P-69",
    #     "HAT-P-70",
    #     "HD 118203",
    #     "HD 1397",
    #     "HD 149026",
    #     "HD 152843",
    #     "HD 189733",
    #     "HD 202772 A",
    #     "HD 209458",
    #     "HD 219666",
    #     "HD 221416",
    #     "HD 332231",
    #     "HD 88133",
    #     "HD 89345",
    #     "K2-232",
    #     "KELT-11",
    #     "KELT-17",
    #     "KELT-19",
    #     "KELT-2 A",
    #     "KELT-20",
    #     "KELT-24",
    #     "KELT-3",
    #     "KELT-4 A",
    #     "KELT-7",
    #     "KOI-13",
    #     "LTT 9779",
    #     "MASCARA-1",
    #     "MASCARA-4",
    #     "TOI-1135",
    #     "TOI-1136",
    #     "TOI-1333",
    #     "TOI-1408",
    #     "TOI-1431",
    #     "TOI-1518",
    #     "TOI-1789",
    #     "TOI-1842",
    #     "TOI-2005",
    #     "TOI-2145",
    #     "TOI-2497",
    #     "TOI-257",
    #     "TOI-421",
    #     "TOI-4551",
    #     "TOI-4603",
    #     "TOI-481",
    #     "TOI-5108",
    #     "TOI-5398",
    #     "TOI-6038 A",
    #     "TOI-622",
    #     "TOI-677",
    #     "TOI-778",
    #     "WASP-131",
    #     "WASP-136",
    #     "WASP-14",
    #     "WASP-166",
    #     "WASP-178",
    #     "WASP-18",
    #     "WASP-33",
    #     "WASP-38",
    #     "WASP-7",
    #     "WASP-74",
    #     "WASP-76",
    #     "WASP-79",
    #     "WASP-8",
    #     "WASP-82",
    #     "WASP-94 A",
    #     "WASP-95",
    #     "WASP-99",
    #     "XO-3",
    #     "WASP-12",
    #     "HAT-P-32",
    #     "Kepler-13",
    #     "TrES-4",
    #     "HD 189733",
    #     "HD 189733 B",
    #     "HAT-P-7",
    # ]

    star_names = ["WASP-82"]
    
    RADIUS_ARCSEC = 450.0 # 10 arcmin
 
    mag_limit = 20.0
    GAIA_USE_ASYNC_JOBS = True

    if existing_output_dir is None:
        output_dir, _, _ = setup_output_directory()
        run_queries = True
    else:
        output_dir = Path(existing_output_dir)
        run_queries = False

    target_rows = []

    if run_queries:
        for name in star_names:
            print(f"Processing {name}")

            try:
                tbl, coord = gaia_nearby_stars_with_params_by_name(name, radius_arcsec=RADIUS_ARCSEC, mag_limit=mag_limit, GAIA_USE_ASYNC_JOBS=GAIA_USE_ASYNC_JOBS)

                if tbl is None or len(tbl) == 0:
                    print(f"No Gaia rows returned for {name}")
                    continue

                background_tbl, target_row = split_target_and_background(tbl, coord)

                if target_row is not None and len(target_row) > 0:
                    target_rows.append((name, target_row))

                filename = name.replace(" ", "_") + ".csv"
                filepath = output_dir / filename

                renamed_background_tbl = rename_output_columns(background_tbl)
                renamed_background_tbl.write(filepath, format="ascii.csv", overwrite=True)

                print(f"Saved {len(background_tbl)} background rows to {filepath}")

            except Exception as e:
                print(f"Failed for {name}: {e}")

        write_targets_excel(target_rows, output_dir)

    report_saved_vs_missing(star_names, output_dir)
    build_master_excel_from_csvs(output_dir)


if __name__ == "__main__":
    main()