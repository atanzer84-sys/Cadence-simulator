import re
import csv
from pathlib import Path
from loaders.run_waltzer_context import setup_output_directory

photometry_pattern = re.compile(
    r"Aperture photometry \((?P<channel>[^)]+)\):.*?"
    r"Cc=(?P<Cc>[-+0-9.eE]+)\s+"
    r"Ca=(?P<Ca>[-+0-9.eE]+)\s+"
    r"Nc=(?P<Nc>\d+)\s+"
    r"Na=(?P<Na>\d+)\s+"
    r"C_background=(?P<C_background>[-+0-9.eE]+)\s+"
    r"C_star=(?P<C_star>[-+0-9.eE]+)\s+"
    r"C_star_noise=(?P<C_star_noise>[-+0-9.eE]+)"
)

target_pattern = re.compile(
    r"UserConfig\(target_name='(?P<target>[^']+)'"
)

exposure_pattern = re.compile(
    r"Channel (?P<channel>\w+) science frame calculation:.*?exposure_s=(?P<exposure>[0-9.]+)"
)

parameter_file_pattern = re.compile(
    r"User parameter file loaded: (?P<parameter_file>.+)"
)


def main(existing_output_dir=None):

    # --- setup output directory ---
    if existing_output_dir is None:
        log_dir, _, _ = setup_output_directory()
    else:
        log_dir = Path(existing_output_dir)

    out_csv = log_dir / "photometry_summary.csv"

    rows = []

    for log_file in sorted(log_dir.rglob("*.log")):

        target_name = None
        exposures = {}
        frame_counters = {}

        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, start=1):

                t = target_pattern.search(line)
                if t:
                    target_name = t.group("target")

                e = exposure_pattern.search(line)
                if e:
                    exposures[e.group("channel")] = float(e.group("exposure"))


                m = photometry_pattern.search(line)
                if not m:
                    continue

                channel = m.group("channel")

                key = (log_file.parent.name, channel)

                if key not in frame_counters:
                    frame_counters[key] = 0

                frame_index = frame_counters[key]
                frame_counters[key] += 1

                rows.append({
                    "run_folder": log_file.parent.name,
                    "target": target_name,
                    "line_number": line_number,
                    "channel": channel,
                    "exposure_s": exposures.get(channel),
                    "frame": frame_index,
                    "C_background": m.group("C_background"),
                    "C_star": float(m.group("C_star")),
                    "C_star_noise": float(m.group("C_star_noise")),
                    "Cc": float(m.group("Cc")),
                    "Ca": float(m.group("Ca")),
                    "Nc": int(m.group("Nc")),
                    "Na": int(m.group("Na")),
                    "C_background": float(m.group("C_background")),
                })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_folder",
                "target",
                "line_number",
                "channel",
                "exposure_s",
                "frame",
                "Cc",
                "Ca",
                "Nc",
                "Na",
                "C_background",
                "C_star",
                "C_star_noise",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    # use existing logs
    main("/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/output")

    # OR fresh run dir
    # main()