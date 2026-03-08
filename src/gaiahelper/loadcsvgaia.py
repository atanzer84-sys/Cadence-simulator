# src/gaiahelper/loadcsvgaia.py

from pathlib import Path
from astropy.table import Table


def collect_results(background_dir: Path, mag_col: str):
    results = []

    for csv_path in sorted(background_dir.glob("*.csv")):
        table = Table.read(csv_path, format="csv")
        if len(table) == 0:
            continue
        if mag_col not in table.colnames:
            continue

        mags = table[mag_col]
        idx = int(mags.argmin())
        brightest_mag = float(mags[idx])
        row = table[idx]

        results.append((csv_path.name, brightest_mag, len(table), row))

    results.sort(key=lambda x: (x[1], -x[2]))
    return results


def print_detailed(results):
    for name, brightest_mag, nrows, row in results:
        print("=" * 80)
        print(f"{name}  brightest_G={brightest_mag:.3f}  rows={nrows}")
        for col in row.colnames:
            print(f"{col}: {row[col]}")


def print_summary(results):
    print()
    print("=" * 80)
    print("BACKGROUND STAR SUMMARY")
    print("=" * 80)
    print(f"{'file':40s} {'n_stars':>10s} {'brightest_G':>12s}")

    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    for name, brightest_mag, nrows, _ in results_sorted:
        print(f"{name:40s} {nrows:10d} {brightest_mag:12.3f}")


def main():
    background_dir = Path(__file__).resolve().parents[2] / "data" / "BackgroundStars"
    mag_col = "phot_g_mean_mag"

    results = collect_results(background_dir, mag_col)

    print_detailed(results)
    print_summary(results)


if __name__ == "__main__":
    main()