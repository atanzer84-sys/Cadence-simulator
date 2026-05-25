from astroquery.gaia import Gaia
from datetime import datetime
from pathlib import Path

from loaders.load_gaia import _gaia_query_for_source_ids
from loaders.run_cadence_context import setup_output_directory


def ts(*args):
    print(datetime.now().strftime('%H:%M:%S'), *args)


def main():
    gaia_ids = [
        4096228993141227776,
        4096228920052984320,
        3521407509481957120,
        4096229130580160000,
        4096229263619726464,
        4096229297981192320,
        4096229199215903872,
        4096229332339210496,
        4096229336738641536,
        4096230122607368960,
        4096229572851298304,
        4096229813369533184,
        4096231604481648256,
        4096232360395929600,
        4096234039617910016,
        4096237102021585152,
        4096243179303476096,
        4096245825006564992,
        4096267643437188096,
        4096281284255675264,
        4096283586358144128,
    ]

    gaia_ids = list(dict.fromkeys(gaia_ids))

    Gaia.TIMEOUT = 120

    ts("Building Gaia query for", len(gaia_ids), "source_ids...")
    query = _gaia_query_for_source_ids(gaia_ids)

    ts("Launching Gaia query...")
    job = Gaia.launch_job_async(query)
    out = job.get_results()

    ts("Rows returned:", len(out))

    output_dir, _, _ = setup_output_directory()
    csv_path = Path(output_dir) / "gaia_targets.csv"

    out.write(csv_path, format="ascii.csv", overwrite=True)

    print(f"Saved CSV file to {csv_path}")


if __name__ == "__main__":
    main()