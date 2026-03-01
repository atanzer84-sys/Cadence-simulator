# src/gaiahelper/loadcsvgaia.py

from pathlib import Path
from astropy.table import Table

def main():
    from pathlib import Path
    from astropy.table import Table

    background_dir = Path(__file__).resolve().parents[2] / "data" / "BackgroundStars"
    mag_col = "phot_g_mean_mag"

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

    for name, brightest_mag, nrows, row in results:
        print("=" * 80)
        print(f"{name}  brightest_G={brightest_mag:.3f}  rows={nrows}")
        for col in row.colnames:
            print(f"{col}: {row[col]}")


if __name__ == "__main__":
    main()