from pathlib import Path
from astropy.table import Table
import numpy as np


def collect_results(background_dir: Path, mag_col: str, mag_limit: float):
    results = []

    for csv_path in sorted(background_dir.glob("*.csv")):
        table = Table.read(csv_path, format="csv")
        if len(table) == 0 or mag_col not in table.colnames:
            continue

        mags = np.asarray(table[mag_col], dtype=float)
        n_stars = len(table)

        count_lt7 = np.sum(mags < 7)
        count_7_8 = np.sum((mags >= 7) & (mags < 8))
        count_8_9 = np.sum((mags >= 8) & (mags < 9))
        count_9_10 = np.sum((mags >= 9) & (mags < 10))
        count_10_11 = np.sum((mags >= 10) & (mags < 11))
        count_11_12 = np.sum((mags >= 11) & (mags < 12))
        count_12_13 = np.sum((mags >= 12) & (mags < 13))
        count_13_14 = np.sum((mags >= 13) & (mags < 14))
        count_14_15 = np.sum((mags >= 14) & (mags < 15))
        count_15_16 = np.sum((mags >= 15) & (mags < 16))
        count_16_17 = np.sum((mags >= 16) & (mags < 17))
        count_17_18 = np.sum((mags >= 17) & (mags < 18))
        count_ge18 = np.sum(mags >= 18)
        count_lt_limit = np.sum(mags < mag_limit)

        results.append((csv_path.name, n_stars, count_lt7, count_7_8, count_8_9, count_9_10, count_10_11, count_11_12, count_12_13, count_13_14, count_14_15, count_15_16, count_16_17, count_17_18, count_ge18, count_lt_limit))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def print_summary(results, mag_limit):
    print()
    print("=" * 190)
    print("BACKGROUND STAR MAGNITUDE SUMMARY")
    print("=" * 190)

    header = f"{'file':35s} {'n_stars':>8s} {'<7':>5s} {'7-8':>5s} {'8-9':>5s} {'9-10':>5s} {'10-11':>6s} {'11-12':>6s} {'12-13':>6s} {'13-14':>6s} {'14-15':>6s} {'15-16':>6s} {'16-17':>6s} {'17-18':>6s} {'>=18':>6s} {('<'+str(mag_limit)):>8s}"
    print(header)
    print("=" * 190)

    for row in results:
        name, n_stars, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c_limit = row
        print(f"{name:35s} {n_stars:8d} {c1:5d} {c2:5d} {c3:5d} {c4:5d} {c5:6d} {c6:6d} {c7:6d} {c8:6d} {c9:6d} {c10:6d} {c11:6d} {c12:6d} {c13:6d} {c_limit:8d}")


def main():
    background_dir = Path(__file__).resolve().parents[2] / "data" / "BackgroundStars"
    mag_col = "phot_g_mean_mag"
    mag_limit = 18

    results = collect_results(background_dir, mag_col, mag_limit)
    print_summary(results, mag_limit)


if __name__ == "__main__":
    main()