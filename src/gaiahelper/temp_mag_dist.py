from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ts(*args):
    print(datetime.now().strftime("%H:%M:%S"), *args)


def combine_csvs(input_dir: str) -> pd.DataFrame:
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_path}")

    ts(f"Found {len(csv_files)} CSV files in {input_path}")

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = csv_file.name
            dfs.append(df)
        except Exception as exc:
            ts(f"Failed to read {csv_file.name}: {exc}")

    if not dfs:
        raise ValueError(f"No readable CSV files in {input_path}")

    return pd.concat(dfs, ignore_index=True)


def plot_temperature_magnitude_distance(
    df: pd.DataFrame,
    temperature_col: str,
    magnitude_col: str,
    distance_col: str,
    title: str,
    output_png: str,
) -> None:
    plot_df = df[[temperature_col, magnitude_col, distance_col]].copy()

    plot_df[temperature_col] = pd.to_numeric(plot_df[temperature_col], errors="coerce")
    plot_df[magnitude_col] = pd.to_numeric(plot_df[magnitude_col], errors="coerce")
    plot_df = plot_df[plot_df[magnitude_col] <= 15]
    plot_df[distance_col] = pd.to_numeric(plot_df[distance_col], errors="coerce")

    plot_df = plot_df.dropna()

    if plot_df.empty:
        ts(f"No valid rows left for plotting: {title}")
        return

    plt.figure(figsize=(16, 10))

    sc = plt.scatter(
        plot_df[distance_col],
        plot_df[magnitude_col],
        c=plot_df[temperature_col],
        s=20,
    )
    plt.xlim(0, 5000)
    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature [K]")

    plt.xlabel("Distance [pc]")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.axhline(y=12)
    plt.text(plt.xlim()[0], 12, "12 mag")
    # sc = plt.scatter(
    #     plot_df[temperature_col],
    #     plot_df[magnitude_col],
    #     c=plot_df[distance_col],
    #     s=10,
    # )

    # cbar = plt.colorbar(sc)
    # cbar.set_label("Distance [pc]")

    # plt.xlabel("Temperature [K]")
    # plt.ylabel("Magnitude")
    # plt.title(title)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    ts(f"Saved plot to {output_png}")


def main(input_dir: str, output_png: str, mode: str) -> None:
    df = combine_csvs(input_dir)

    if mode == "vband":
        plot_temperature_magnitude_distance(
            df=df,
            temperature_col="Teff",
            magnitude_col="Vmag",
            distance_col="Dist",
            title="All V-band stars",
            output_png=output_png,
        )
    elif mode == "background":
        plot_temperature_magnitude_distance(
            df=df,
            temperature_col="Teff",
            magnitude_col="phot_g_mean_mag",
            distance_col="dist_pc",
            title="All background stars",
            output_png=output_png,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main(
        input_dir="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/vband",
        output_png="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/all_vband5800_stars.png",
        mode="vband",
    )
    main(
        input_dir="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/v3500",
        output_png="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/all_v3500_stars.png",
        mode="vband",
    )

    main(
        input_dir="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/back",
        output_png="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/all_background_stars.png",
        mode="background",
    )
    main(
        input_dir="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/all",
        output_png="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/all_stars.png",
        mode="background",
    )
    main(
        input_dir="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/kelt20",
        output_png="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output/kelt20.png",
        mode="background",
    )
