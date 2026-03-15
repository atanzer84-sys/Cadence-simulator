from datetime import datetime
from pathlib import Path
import pandas as pd

def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)


def build_master_excel_from_csvs(output_dir):
    csv_files = sorted(Path(output_dir).glob("*.csv"))

    if not csv_files:
        print("No CSV files found for Excel export.")
        return

    all_dfs = []

    for csv_file in csv_files:
        # Read all columns as strings so large integer IDs (e.g. GAIA, TIC)
        # are not rounded to float64 during CSV parsing.
        df = pd.read_csv(csv_file, dtype=str)
        target_star = csv_file.stem.replace("_", " ")
        df.insert(0, "target_star", target_star)
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Convert separation column back to numeric for proper sorting, if present.
    if "sep_arcsec" in master_df.columns:
        master_df["sep_arcsec"] = pd.to_numeric(master_df["sep_arcsec"], errors="coerce")
        master_df = master_df.sort_values(["target_star", "sep_arcsec"], ascending=[True, True])

    # Ensure ID-like columns are preserved exactly when opened in Excel
    for col in ("source_id", "TIC", "GAIA"):
        if col in master_df.columns:
            master_df[col] = master_df[col].astype(str)

    # German decimal separator: write numeric columns as text with "," so Excel shows them correctly
    _TEXT_COLUMNS = {"target_star", "source_id", "TIC", "GAIA", "LClass", "is_target"}
    for col in master_df.columns:
        if col in _TEXT_COLUMNS:
            continue
        numeric = pd.to_numeric(master_df[col], errors="coerce")
        if numeric.notna().any():
            master_df[col] = numeric.apply(lambda x: "" if pd.isna(x) else str(x).replace(".", ","))

    excel_path = Path(output_dir) / "stars.xlsx"
    master_df.to_excel(excel_path, index=False)

    print(f"Saved master Excel file to {excel_path}")



def main(existing_output_dir):

    output_dir = Path(existing_output_dir)
    build_master_excel_from_csvs(output_dir)

if __name__ == "__main__":
    main("/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/Faint")