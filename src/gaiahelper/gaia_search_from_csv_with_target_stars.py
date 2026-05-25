from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from datetime import datetime
from loaders.run_cadence_context import setup_output_directory
from pathlib import Path
import astropy.units as u
import pandas as pd


def ts(*args):
    print(datetime.now().strftime("%H:%M:%S"), *args)


INPUT_CSV = "/Users/andreatanzer/Documents/Space Science/MasterThesis/waltzerclone/stars.csv"

TARGET_NAME_COL = "hostname"
RA_COL = "ra"
DEC_COL = "dec"
TARGET_MAG_COL = "sy_vmag"

RADIUS_ARCSEC = 100.0
MAG_LIMIT = 13.0

def gaia_nearby_stars_with_params_by_coord(ra_deg: float, dec_deg: float, radius_arcsec: float, mag_limit: float):
    Gaia.TIMEOUT = 120

    radius_deg = radius_arcsec / 3600.0

    ts("Preparing Gaia query", "RA=", ra_deg, "DEC=", dec_deg)

    query = f"""
    SELECT
        gs.source_id,
        gs.ra,
        gs.dec,
        gs.parallax,
        gs.phot_g_mean_mag,
    FROM gaiadr3.gaia_source AS gs
    WHERE 1 = CONTAINS(
        POINT('ICRS', gs.ra, gs.dec),
        CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
    )
    AND gs.phot_g_mean_mag < {mag_limit}
    """

    ts("Launching Gaia query...")
    job = Gaia.launch_job_async(query)

    ts("Fetching Gaia results...")
    out = job.get_results()

    ts("Gaia rows returned:", len(out))

    if len(out) == 0:
        return out

    out.sort("phot_g_mean_mag")
    ts("Sorted Gaia results by phot_g_mean_mag")

    return out


def load_target_stars_from_csv(csv_path: str | Path):
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    required_cols = [TARGET_NAME_COL, RA_COL, DEC_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    target_rows = []
    for _, row in df.iterrows():
        target_name = str(row[TARGET_NAME_COL]).strip()
        ra_deg = float(row[RA_COL])
        dec_deg = float(row[DEC_COL])

        target_mag = None
        if TARGET_MAG_COL in df.columns and pd.notna(row[TARGET_MAG_COL]):
            target_mag = float(row[TARGET_MAG_COL])

        target_rows.append({
            "target_name": target_name,
            "ra": ra_deg,
            "dec": dec_deg,
            "target_mag": target_mag,
        })

    return target_rows


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).replace(" ", "_")


def split_target_and_background(tbl, target_name: str, target_ra: float, target_dec: float):
    if len(tbl) == 0:
        return None, tbl

    coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg, frame="icrs")
    sources_coord = SkyCoord(ra=tbl["ra"], dec=tbl["dec"], unit="deg", frame="icrs")
    separations = coord.separation(sources_coord)
    tbl["sep_arcsec"] = separations.arcsec

    idx_target = separations.argmin()
    target_row = tbl[idx_target]

    target_record = {
        "target_name": target_name,
        "source_id": str(target_row["source_id"]) if pd.notna(target_row["source_id"]) else None,
        "ra": float(target_row["ra"]) if pd.notna(target_row["ra"]) else None,
        "dec": float(target_row["dec"]) if pd.notna(target_row["dec"]) else None,
        "parallax": float(target_row["parallax"]) if  pd.notna(target_row["parallax"]) else None,
        "phot_g_mean_mag": float(target_row["phot_g_mean_mag"]) if pd.notna(target_row["phot_g_mean_mag"]) else None,
        "Teff": float(target_row["Teff"]) if pd.notna(target_row["Teff"]) else None,
        "dist_pc": float(target_row["dist_pc"]) if pd.notna(target_row["dist_pc"]) else None,
    }

    tbl.remove_row(idx_target)

    if len(tbl) > 0:
        tbl.sort("sep_arcsec")

    return target_record, tbl


def write_background_stars_csv(tbl, output_dir: Path, target_name: str):
    filename = sanitize_filename(target_name) + ".csv"
    filepath = output_dir / filename
    tbl.write(filepath, format="ascii.csv", overwrite=True)
    print(f"Saved {len(tbl)} background stars to {filepath}")


def write_master_targets_csv(target_rows, output_dir: Path):
    master_df = pd.DataFrame(target_rows)

    if not master_df.empty:
        master_df = master_df.sort_values("target_name")

    if "source_id" in master_df.columns:
        master_df["source_id"] = master_df["source_id"].astype(str)

    filepath = output_dir / "all_target_stars.csv"
    master_df.to_csv(filepath, index=False)

    print(f"Saved {len(master_df)} target stars to {filepath}")


def main(existing_output_dir=None):
    if existing_output_dir is None:
        output_dir, _, _ = setup_output_directory()
    else:
        output_dir = Path(existing_output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = load_target_stars_from_csv(INPUT_CSV)
    master_target_rows = []

    for i, target in enumerate(target_rows, start=1):
        target_name = target["target_name"]
        target_ra = target["ra"]
        target_dec = target["dec"]

        print(f"[{i}/{len(target_rows)}] Processing {target_name}")

        try:
            ts("Starting Gaia query for", target_name)

            tbl = gaia_nearby_stars_with_params_by_coord(
                ra_deg=target_ra,
                dec_deg=target_dec,
                radius_arcsec=RADIUS_ARCSEC,
                mag_limit=MAG_LIMIT,
            )

            ts("Splitting target and background for", target_name)
            target_record, background_tbl = split_target_and_background(
                tbl=tbl,
                target_name=target_name,
                target_ra=target_ra,
                target_dec=target_dec,
            )

            if target_record is not None:
                master_target_rows.append(target_record)
            else:
                master_target_rows.append({
                    "target_name": target_name,
                    "source_id": None,
                    "ra": None,
                    "dec": None,
                    "parallax": None,
                    "phot_g_mean_mag": None,
                    "Teff": None,
                    "dist_pc": None,
                })

            ts("Writing background CSV for", target_name)
            write_background_stars_csv(background_tbl, output_dir, target_name)

            ts("Done with", target_name)

        except Exception as e:
            print(f"Failed for {target_name}: {e}")
            master_target_rows.append({
                "target_name": target_name,
                "source_id": None,
                "ra": None,
                "dec": None,
                "parallax": None,
                "phot_g_mean_mag": None,
                "Teff": None,
                "dist_pc": None,
            })

    write_master_targets_csv(master_target_rows, output_dir)


if __name__ == "__main__":
    # main("/Users/andreatanzer/Documents/Space Science/MasterThesis/waltzerclone/WALTzER-simulator/output/test")
    main()