"""Spectral modelling: stellar atmosphere to photon flux density at the observer.

This module implements run_photon_flux_density_pipeline, which goes from a 1D temperature
model through geometric dilution, optional line-core and ISM steps, reddening correction
(unred), and finally conversion to spectral photon flux density. The input star may be
the science target or a background field star; the caller passes background_star when
dropping some optional diagnostics.

Typical use from the simulator is instrument.prepare_detector_images.calculate_photon_flux_density_on_Earth,
which chooses wl_min_A / wl_max_A from the enabled channel configs and then calls
run_photon_flux_density_pipeline.

Outputs are aligned float32 arrays: photon flux density in photons per second per cm² per
angstrom, and wavelength in angstroms. Intermediate products written as FluxCalc_* dumps
are still in cgs-style spectral energy flux where the filenames indicate that; the
pipeline converts unreddened erg s⁻¹ cm⁻² Å⁻¹ to photons using convert_flux_to_photons.
"""
import logging
import numpy as np
from domain.star import Star
from utils.constants import C_LIGHT_Angst, PARSEC_CM, R_V, PHOTON_ENERGY_CONVERSION_A
from configs.global_config import get_global_config, GlobalConfig
from flux.cute_line_core_emission import apply_line_core_emission
from flux.cute_extinction import extinction_amores
from flux.cute_ism_abs_all import cute_ism_abs_all
from flux.cute_unred import unred
from astropy.coordinates import SkyCoord
from astropy import units as u
from loaders.run_cadence_context import RunContext
from loaders.load_model_temperature import load_model_for_temperature
from utils.debug_dumps import dump_1d_array, dump_3d_array, dump_npz_snapshot
from utils.helpers import announce
from utils.flux_image_array import plot_flux_and_photons_windows, plot_model_input


def convert_flux_to_photons(flux_unred, wavelengths):
    """ Converts unreddened f_lambda to photon flux per unit wavelength interval.

    Parameters
    ----------
    flux_unred : array-like
        Spectral energy flux F_λ at the observer, erg s⁻¹ cm⁻² Å⁻¹. Must be
        compatible with ``wavelengths``.
    wavelengths : array-like
        Wavelength λ in angstroms (Å), same shape as ``flux_unred``.

    Returns
    -------
    ndarray
        Spectral photon flux density: photons s⁻¹ cm⁻² Å⁻¹.

    Notes
    -----
    With wavelength in Å, ``photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths``.
    Here ``PHOTON_ENERGY_CONVERSION_A = 1 / (H_PLANCK * C_LIGHT_Angst)`` (see ``utils.constants``):
    this is the usual F_λ → photon flux per Å relation using E_photon = hc/λ.
    """
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths  # from ergs/s/cm2/A to photons/s/cm2/A
    return photon_flux


def run_photon_flux_density_pipeline(
    star: Star, ctx: RunContext, wl_min_A: float, wl_max_A: float, announce_user: bool = False, background_star: bool = False
):
    """Return photon flux density at the observer and the wavelength grid (both float32).

    Runs the per-wavelength pipeline (model, extinction, unred), then converts to photon
    flux density. The first array is photons s⁻¹ cm⁻² Å⁻¹; the second is wavelength in Å;
    same length.
    """
    announce(f"Starting photon flux density pipeline for star {star.name}", announce_user)
    cfg = get_global_config()
    dump_arrays = cfg.write_intermediate_arrays and not background_star
    dump_plots = cfg.produce_flux_convolution_plots and not background_star

    model_data = load_model_for_temperature(star.effective_temperature, wl_min_A, wl_max_A, announce_user=announce_user)

    if dump_arrays:
        dump_3d_array(model_data, ctx.output_dir, star.name, "FluxCalc_1_model_input", perChannel=True, zoom=True)
    if not background_star and dump_plots:
        plot_model_input(model_data, wl_min_A, wl_max_A, ctx.output_dir, star, filename_tag="FluxCalc_1_model_input", title_text="Model Input Spectrum")
        plot_flux_and_photons_windows(model_data[:, 0], model_data[:, 1], ctx.output_dir, star, "FluxCalc_1_model_input", "Model Input Spectrum (linear windows)", "Model flux [erg s⁻¹ cm⁻² Hz⁻¹]")

    flux_lambda_original = convert_stellar_model_to_flux(model_data, star.radius_sun_cm)
    if not background_star and dump_plots:
        plot_model_input(flux_lambda_original, wl_min_A, wl_max_A, ctx.output_dir, star, filename_tag="FluxCalc_2_convert_stellar_model_to_flux", title_text="Flux After Stellar Model Conversion")

    # keep undiluted flux
    flux_lambda_diluted = flux_lambda_original.copy()
    wavelengths = flux_lambda_original[:, 0]

    if dump_arrays:
        dump_3d_array(flux_lambda_original, ctx.output_dir, star.name, "FluxCalc_2_convertIntensityToLuminosity_snapshot", perChannel=True, zoom=True)

    if cfg.line_core_emission:
        # ACTUAL LCA EMISSION
        flux_lambda_diluted = apply_line_core_emission(flux_lambda_diluted, cfg.sigmaMg22, cfg.sigmaMg21, star.log_r, star.spectral_type, announce_user=announce_user)

        if dump_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_3_after_line_core_emission", perChannel=True, zoom=True)

    else:
        logging.info("Line Core Emission not applied!")

    # Starting extinction
    ebv, _ = compute_ebv_av(star.right_ascension, star.declination, star.distance_pc)

    if cfg.interstellar_absorption:
        # ACTUAL ISM_ABS CALL
        flux_lambda_diluted = apply_ism_absorption(flux_lambda_diluted, ebv, cfg, announce_user=announce_user)

        if dump_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_4_after_ISM", perChannel=True, zoom=True)

    else:
        logging.info("Interstellar Medium absorption not applied!")


    flux_di_before = flux_lambda_diluted[:, 1].copy()
    if dump_arrays:
        dump_1d_array(wavelengths, flux_di_before, ctx.output_dir, star.name, "FluxCalc_5_before_flux_at_earth", perChannel=True, zoom=True)

    # FINALLY FLUX ON EARTH
    flux_at_earth = compute_flux_at_earth(flux_lambda_diluted, star.distance_pc, announce_user=announce_user)

    if dump_arrays:
        dump_1d_array(wavelengths, flux_at_earth, ctx.output_dir, star.name, "FluxCalc_6_after_flux_at_earth", perChannel=True, zoom=True)

    # UNRED FLUX
    flux_unred = apply_unred(wavelengths, flux_at_earth, ebv, announce_user=announce_user)

    if dump_arrays:
        dump_1d_array(wavelengths, flux_unred, ctx.output_dir, star.name, "FluxCalc_7_after_unred", perChannel=True, zoom=True)
    if dump_plots:
        plot_flux_and_photons_windows(wavelengths, flux_unred, ctx.output_dir, star, "FluxCalc_1_Flux", "Flux", "Flux [erg s⁻¹ cm⁻² Å⁻¹]")

    photon_flux = convert_flux_to_photons(flux_unred, wavelengths)
    photon_flux = np.asarray(photon_flux, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)

    if dump_arrays:
        dump_1d_array(wavelengths, photon_flux, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_FluxCalc_8_photons_star_full.npz", photons_star=photon_flux, wavelengths=wavelengths)

    if dump_plots:
        plot_flux_and_photons_windows(wavelengths, photon_flux, ctx.output_dir, star, "FluxCalc_photons", "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")

    return photon_flux, wavelengths


def convert_stellar_model_to_flux(model_data, r_star):
    """Map the loaded stellar grid from F_ν at the surface to L_λ (spectral luminosity).

    The model file supplies wavelength and surface flux density per unit frequency.
    This routine applies F_ν → F_λ (using c in Å/s), multiplies by the emitting area
    4πR², and by 4π steradians so the result is integrated stellar spectral luminosity
    in two parallel columns (same units as each other).

    Parameters
    ----------
    model_data : ndarray, shape (n, 3)
        Column 0: wavelength λ in angstroms (Å).
        Columns 1 and 2: surface flux density F_ν in erg cm⁻² s⁻¹ Hz⁻¹ (two components,
        e.g. from the atmosphere grid).
    r_star : float
        Stellar radius in cm.

    Returns
    -------
    flux_lambda : ndarray, shape (n, 3)
        Column 0: λ (Å), copied from the input.
        Columns 1 and 2: spectral luminosity L_λ in erg s⁻¹ Å⁻¹ per component (integrated
        over the stellar surface and full solid angle as encoded in the scale factor).

    Notes
    -----
    Uses F_λ = F_ν · c / λ² with c = C_LIGHT_Angst (Å/s). The factor
    ``C_LIGHT_Angst * 4π R² * 4π / λ²`` matches the existing implementation of surface
    area and steradian bookkeeping described in the inline comments below.
    """
    flux_lambda = np.zeros(np.shape(model_data))
    wavelength = model_data[:, 0]
    wavelength_sq = wavelength * wavelength
    scale = C_LIGHT_Angst * 4 * np.pi * (r_star**2) * 4 * np.pi

    # F_nu (erg/cm^2/s/Hz) -> F_lambda (erg/cm^2/s/A) on the surface: F_lambda = F_nu * c / lambda^2
    # Then: surface flux (erg/cm^2/s/A) * (4*pi*R^2) cm^2 -> erg/s/A emitted over the disk;
    # the extra 4*pi is the legacy solid-angle convention used here per original comments.
    flux_lambda[:, 0] = wavelength
    flux_lambda[:, 1] = model_data[:, 1] * scale / wavelength_sq
    flux_lambda[:, 2] = model_data[:, 2] * scale / wavelength_sq

    logging.info("Converting intensity to luminosity for r_star=%.6e cm with %d wavelength points", r_star, model_data.shape[0])

    return flux_lambda

def apply_ism_absorption(data, ebv, cfg: GlobalConfig, announce_user: bool = False):
    announce("Starting ISM absorption", announce_user)
    if cfg.mg2_col is None:
        nh = 5.8e21 * ebv  # is the total hydrogen column density
        fractionMg2 = 0.825  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        # Safe runtime path: keep physically correct -inf for zero columns without triggering log10(0) warnings.
        mg2_col_linear = nh * fractionMg2 * 10.0**Mg_abn
        nmg2 = -np.inf if mg2_col_linear <= 0.0 else np.log10(mg2_col_linear)
        nmg2_source = "computed"
    else:
        nmg2 = float(cfg.mg2_col)
        nmg2_source = "cfg"

    if cfg.mg1_col is None:
        nh = 5.8e21 * ebv   # is the total hydrogen column density
        fractionMg1 = 0.00214  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        mg1_col_linear = nh * fractionMg1 * 10.0**Mg_abn
        nmg1 = -np.inf if mg1_col_linear <= 0.0 else np.log10(mg1_col_linear)
        nmg1_source = "computed"
    else:
        nmg1 = float(cfg.mg1_col)
        nmg1_source = "cfg"

    if cfg.fe2_col is None:
        nh = 5.8e21 * ebv   # is the total hydrogen column density
        fractionFe2 = 0.967  # (Frisch & Slavin 2003; this is the fraction of Fe in the ISM that is singly ionised)
        Fe_abn = -5.73  # (Frisch & Slavin 2003; this is the ISM abundance of Fe)
        fe2_col_linear = nh * fractionFe2 * 10.0**Fe_abn
        nfe2 = -np.inf if fe2_col_linear <= 0.0 else np.log10(fe2_col_linear)
        nfe2_source = "computed"
    else:
        nfe2 = float(cfg.fe2_col)
        nfe2_source = "cfg"

    logging.info("ISM params: E(B-V)=%s nmg2=%s (%s) nmg1=%s (%s) nfe2=%s (%s)", ebv, nmg2, nmg2_source, nmg1, nmg1_source, nfe2, nfe2_source)

    flux_data = cute_ism_abs_all(data, nmg2, nmg1, nfe2)

    return flux_data

def compute_ebv_av(right_ascension, declination, distance_pc):
    distance_kpc = distance_pc / 1000.0
    glon, glat = calculate_glon_glat(right_ascension, declination)
    ebv, av = extinction_amores(glon, glat, distance_kpc)
    logging.info("EBV=%s AV=%s (glon=%s glat=%s dist_kpc=%s)", ebv, av, glon, glat, distance_kpc)

    return ebv, av

def calculate_glon_glat(right_ascension, declination):
    c = SkyCoord(ra=right_ascension, dec=declination, unit=(u.degree, u.degree))
    glon = c.galactic.l.deg
    glat = c.galactic.b.deg
    return glon, glat

def compute_flux_at_earth(flux_lambda_diluted, distance_pc, announce_user: bool = False):
    """Geometric dilution: spectral luminosity to spectral flux at Earth.

    Applies the inverse-square law F_λ = L_λ / (4π d²) using the second column of the
    diluted luminosity array and the star–observer distance in parsecs.

    Parameters
    ----------
    flux_lambda_diluted : ndarray, shape (n, at least 2)
        Column 0: wavelength (Å), unused here.
        Column 1: spectral luminosity L_λ in erg s⁻¹ Å⁻¹ (same convention as after
        line-core / ISM processing in the pipeline).
    distance_pc : float
        Distance to the star in parsecs (pc).
    announce_user : bool
        Passed to ``announce`` for optional progress text.

    Returns
    -------
    ndarray, shape (n,)
        Spectral energy flux at Earth, F_λ in erg s⁻¹ cm⁻² Å⁻¹, one value per row.
    """
    announce("Starting Flux at Earth calculation", announce_user)
    flux_di = flux_lambda_diluted[:,1]
    flux_at_earth = flux_di / (4.0 * np.pi * (distance_pc * PARSEC_CM) ** 2)
    return flux_at_earth

def apply_unred(wavelengths, flux_at_earth, ebv, announce_user: bool = False):
    announce("Starting to apply UNRED extinction correction", announce_user)
    ebv = -1.0 * ebv
    flux_unred = unred(wavelengths, flux_at_earth, ebv=ebv, R_V=R_V)
    return flux_unred
