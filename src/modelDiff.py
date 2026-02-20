import numpy as np
import matplotlib.pyplot as plt

from loaders.run_setup import setup_output_directory
from flux.flux_calc import load_model_for_temperature


def main() -> None:
    teff = 5750.0
    wl_min = 1000.0
    wl_max = 19000.0

    output_dir, _ = setup_output_directory()

    model = load_model_for_temperature(teff)
    wl_full = np.asarray(model[:, 0], dtype=np.float64)

    if wl_full.size < 2:
        raise ValueError("Wavelength array too short.")

    # ----- REGIMES ON FULL GRID -----
    delta_full = wl_full[1:] - wl_full[:-1]

    decimals = 6
    d = np.round(delta_full, decimals)

    start = 0
    prev = d[0]

    print("Changes in wavelength grid (wl_start, wl_end, delta):")
    for i in range(1, d.size):
        if d[i] != prev:
            print(float(wl_full[start]), float(wl_full[i]), float(prev))
            start = i
            prev = d[i]
    print(float(wl_full[start]), float(wl_full[-1]), float(prev))

    # ----- PLOT ONLY 1000..19000 -----
    mask = (wl_full >= wl_min) & (wl_full <= wl_max)
    wl = wl_full[mask]

    delta = wl[1:] - wl[:-1]
    x = wl[:-1]
    y = delta

    fig = plt.figure()
    plt.plot(x, y, color="green", linewidth=2.5)
    plt.xlabel("Wavelength")
    plt.ylabel("Wavelength grid delta in A")
    plt.xlim(wl_min, wl_max)
    plt.ylim(0.0, 0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "wavelength_diffs_1000_19000.png", dpi=200)
    plt.close(fig)

    print("Plot written to:", output_dir / "wavelength_diffs_1000_19000.png")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()