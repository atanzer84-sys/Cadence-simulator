import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from astropy.table import Table
from loaders.run_waltzer_context import get_repo_root
from configs.global_config import GlobalConfig
from domain.star_catalog import StarCatalog
from loaders.load_stellar_and_planetary_properties import load_excel_mapping , infer_mamajek, apply_log_r_fallback, get_missing_properties
from loaders.load_gaia import get_gaia_stellar_properties
from flux.flux_calc import calculateFluxOnEarth
from loaders.load_gaia import gaia_lookup_for_background_stars

def lookup_background_stars(ctx: RunContext, cfg: GlobalConfig, star: Star):
    table = load_background_csv_if_exists(star)
    if table is None:
        table = gaia_lookup_for_background_stars(star, g_mag_limit=cfg.magnitude_cutoff, GAIA_USE_ASYNC_JOBS=cfg.GAIA_USE_ASYNC_JOBS)
        if table is not None and len(table) > 0:
            save_background_stars_csv(table, ctx.output_dir, star.name)
    if table is None or len(table) == 0:
        return StarCatalog()

    catalog = create_background_star_catalog(table, cfg)

    total = len(catalog.stars_by_id)
    logging.info("Starting flux calculation for %d background stars", total)
    print(f"==== STARTING FLUX CALCULATION FOR {total} BACKGROUND STARS =====")
    
    for i, (star_id, bg_star) in enumerate(catalog.stars_by_id.items(), start=1):
        logging.info("Calculating Flux on Earth %d/%d for %s", i, total, star_id)

        print(f"Flux {i}/{total} for {star_id}")
        flux_unred, wavelengths = calculateFluxOnEarth(bg_star, ctx)
        catalog.flux_earth_by_id[star_id] = (wavelengths, flux_unred)    

    return catalog
def load_background_csv_if_exists(star: Star) -> Table | None:
    repo_root = get_repo_root()
    csv_name = star.name.replace(" ", "_")
    csv_path = repo_root / "data" / "BackgroundStars" / f"{csv_name}.csv"

    if not csv_path.exists():
        return None

    table = Table.read(csv_path, format="csv")

    logging.info("Background stars: loading cached CSV: %s", csv_path)
    logging.info("Loaded background CSV for %s: rows=%d, columns=%s", star.name, len(table), list(table.colnames))
    return table

def create_background_star_catalog(table, cfg:GlobalConfig):
    catalog = StarCatalog()
    required_keys = load_required_stellar_parameters()

    for row in table:
        star_params = get_gaia_stellar_properties(row, log_output=False)
        if not _passes_magnitude_cutoff(star_params, max_mag=cfg.magnitude_cutoff):
            continue

        _set_background_star_name(star_params, row)
        _apply_distance_from_parallax_if_missing(star_params, row)

        if star_params.get("effective_temperature") is not None:
            star_params = infer_mamajek(star_params, log_output=False)
        
        if star_params.get("radius") is not None:
            star_params = apply_log_r_fallback(star_params, cfg, log_output=False)

        if not _ensure_required_properties(star_params, required_keys):
            continue

        bg_star = Star.from_params(star_params, required_keys, log_output = False)

        catalog.add_star(bg_star.name, bg_star)

    return catalog

def load_required_stellar_parameters():
    mapping = load_excel_mapping()
    return mapping["required_stellar_parameters"]


def save_background_stars_csv(table: Table, output_dir, star_name: str) -> None:
    """Write background stars table to CSV in output_dir. Use same name as cache so it can be moved to data/BackgroundStars/."""
    csv_name = star_name.replace(" ", "_")
    csv_path = output_dir / f"{csv_name}.csv"
    table.write(csv_path, format="ascii.csv", overwrite=True)
    msg = f"Background stars: saved to {csv_path} (move to data/BackgroundStars/ for cache)"
    logging.info(msg)
    print(msg)


def _passes_magnitude_cutoff(star_params: dict, max_mag: float = 20.0) -> bool:
    """True if star has gaia_magnitude and it is <= max_mag."""
    mag = star_params.get("gaia_magnitude")
    if mag is None:
        return False
    return float(mag) <= max_mag


def _set_background_star_name(star_params: dict, row) -> None:
    """Set star_params['name'] from row source_id, or '0000' if missing."""
    if "source_id" in row.colnames and row["source_id"] is not None:
        star_params["name"] = f"gaia_{int(row['source_id'])}"
    else:
        star_params["name"] = "0000"


def _apply_distance_from_parallax_if_missing(star_params: dict, row) -> None:
    """If star_params has no distance, set it from row parallax (distance_pc = 1000/parallax_mas) when valid."""
    if star_params.get("distance") is not None:
        return
    if "parallax" not in row.colnames:
        return
    par = row["parallax"]
    if par is None or np.ma.is_masked(par) or not np.isfinite(par) or float(par) <= 0.0:
        return
    star_params["distance"] = 1000.0 / float(par)


def _ensure_required_properties(star_params: dict, required_keys: list[str]) -> bool:
    """Return False if any required keys are missing (logs and skip). Otherwise normalize distance key and return True."""
    missing = get_missing_properties(star_params, required_keys, log_output=False)
    if missing:
        logging.info("Background star %s skipped. Missing: %s", star_params.get("name"), missing)
        return False
    if "distance" in star_params:
        star_params["distance"] = star_params.pop("distance")
    return True






