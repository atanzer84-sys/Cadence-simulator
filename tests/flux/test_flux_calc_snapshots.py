import numpy as np
from pathlib import Path
from flux.flux_calc import convertStellarModelToFlux, apply_line_core_emission, apply_ism_absorption, compute_flux_at_earth, apply_unred
from instrument.prepare_detector_images import convert_flux_to_photons

from utils.constants import R_SUN_cm
from types import SimpleNamespace

SIGMA_MG22 = 0.257
SIGMA_MG21 = 0.288

def make_cfg_for_ism_absorption():
    return SimpleNamespace(
        mg2_col=None,
        mg1_col=None,
        fe2_col=None,
        sigmaMg22=SIGMA_MG22,
        sigmaMg21=SIGMA_MG21,
    )
SNAPSHOT_BASE = Path(__file__).parent / "snapshots"

STARS = {
    "WASP-69": {
        "name": "WASP-69",
        "spectral_type": "K3V",
        "effective_temperature": 4792.0,
        "radius_rsun": 0.801,
        "mass_msun": 0.83,
        "metallicity": 0.35,
        "surface_gravity": 4.57,
        "right_ascension": 315.0259661,
        "declination": -5.094857,
        "distance_pc": 49.9605,
        "v_magnitude": 9.873,
        "gaia_magnitude": 9.48466,
        "log_r": -5.0,
        "radius_sun_cm": 55725570000.0,
        "mass_sun_kg": 1.6504300999999998e+30,
        "EBV": 0.0,
        "AV": 0.0,
    },
    "WASP-189": {
        "name": "WASP-189",
        "spectral_type": "A6V",
        "effective_temperature": 8000.0,
        "radius_rsun": 2.36,
        "mass_msun": 2.03,
        "metallicity": 0.29,
        "surface_gravity": 3.9,
        "right_ascension": 225.686732,
        "declination": -3.0314874,
        "distance_pc": 99.731,
        "v_magnitude": 6.59776,
        "gaia_magnitude": 6.55369,
        "log_r": -7.0,
        "radius_sun_cm": 164185200000.0,
        "mass_sun_kg": 4.0365940999999997e+30,
        "EBV": 0.0,
        "AV": 0.0,
    },
        "HD 2685" : {
        "name": "HD 2685",
        "spectral_type": "F2V",
        "effective_temperature": 6801.0,
        "radius_rsun": 1.56,
        "mass_msun": 1.43,
        "metallicity": 0.02,
        "surface_gravity": 4.21,
        "right_ascension": 7.3289156,
        "declination": -76.304032,
        "distance_pc": 196.852,
        "v_magnitude": 9.595,
        "gaia_magnitude": 9.52025,
        "log_r": -7.0,
        "radius_sun_cm": 108529200000.0,
        "mass_sun_kg": 2.8435121e+30,
        "EBV": 0.017036739005112034,
        "AV": 0.0519620539655917,
    }
}


def run_snapshot_convertIntensityToLuminosity(star_name, radius_rsun, band):
    base = SNAPSHOT_BASE
    model_file = base / f"{star_name}_FluxCalc_1_model_input_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_2_convertIntensityToLuminosity_snapshot_{band}_zoom.txt"
    model = np.loadtxt(model_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)
    got = convertStellarModelToFlux(model, radius_rsun * R_SUN_cm)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def run_snapshot_apply_line_core_emission(star_name, band, sigmaMg22, sigmaMg21, log_r, spectral_type):
    base = SNAPSHOT_BASE
    before_file = base / f"{star_name}_FluxCalc_2_convertIntensityToLuminosity_snapshot_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_3_after_line_core_emission_{band}_zoom.txt"
    before = np.loadtxt(before_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)
    got = apply_line_core_emission(before, sigmaMg22, sigmaMg21, log_r, spectral_type)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def run_snapshot_apply_ism_absorption(star_name, band, EBV):
    base = SNAPSHOT_BASE
    before_file = base / f"{star_name}_FluxCalc_3_after_line_core_emission_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_4_after_ISM_{band}_zoom.txt"
    before = np.loadtxt(before_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)
    cfg = make_cfg_for_ism_absorption()
    got = apply_ism_absorption(before, EBV, cfg)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def run_snapshot_compute_flux_at_earth(star_name, band, distance_pc):
    base = SNAPSHOT_BASE
    before_file = base / f"{star_name}_FluxCalc_5_before_flux_at_earth_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_6_after_flux_at_earth_{band}_zoom.txt"
    before_2d = np.loadtxt(before_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)

    before_3d = np.column_stack((before_2d[:, 0], before_2d[:, 1], np.zeros(before_2d.shape[0], dtype=np.float32)))
    flux = compute_flux_at_earth(before_3d, distance_pc)

    out = np.column_stack((before_2d[:, 0], flux))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0.0)


def run_snapshot_apply_unred(star_name, band):
    base = SNAPSHOT_BASE
    before_file = base / f"{star_name}_FluxCalc_6_after_flux_at_earth_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_7_after_unred_{band}_zoom.txt"
    before_2d = np.loadtxt(before_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)
    flux = apply_unred(before_2d[:, 0], before_2d[:, 1], STARS[star_name]["EBV"])
    out = np.column_stack((before_2d[:, 0], flux))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0.0)

def run_snapshot_convert_flux_to_photons(star_name, band):
    # Verifies convert_flux_to_photons produces the expected photon flux snapshot from after_unred input.
    base = SNAPSHOT_BASE
    before_file = base / f"{star_name}_FluxCalc_7_after_unred_{band}_zoom.txt"
    expected_file = base / f"{star_name}_FluxCalc_8_photons_star_{band}_zoom.txt"

    before_2d = np.loadtxt(before_file, dtype=np.float32)
    expected = np.loadtxt(expected_file, dtype=np.float32)

    wavelengths = before_2d[:, 0]
    flux_unred = before_2d[:, 1]

    photons_star = convert_flux_to_photons(flux_unred, wavelengths)
    out = np.column_stack((wavelengths, photons_star))

    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0.0)

def test_WASP189_convertIntensityToLuminosity_IR():
    run_snapshot_convertIntensityToLuminosity("WASP-189", STARS["WASP-189"]["radius_rsun"], "IR")

def test_WASP189_convertIntensityToLuminosity_NUV():
    run_snapshot_convertIntensityToLuminosity("WASP-189", STARS["WASP-189"]["radius_rsun"], "NUV")

def test_WASP189_convertIntensityToLuminosity_VIS():
    run_snapshot_convertIntensityToLuminosity("WASP-189", STARS["WASP-189"]["radius_rsun"], "VIS")

def test_WASP69_convertIntensityToLuminosity_IR():
    run_snapshot_convertIntensityToLuminosity("WASP-69", STARS["WASP-69"]["radius_rsun"], "IR")

def test_WASP69_convertIntensityToLuminosity_NUV():
    run_snapshot_convertIntensityToLuminosity("WASP-69", STARS["WASP-69"]["radius_rsun"], "NUV")

def test_WASP69_convertIntensityToLuminosity_VIS():
    run_snapshot_convertIntensityToLuminosity("WASP-69", STARS["WASP-69"]["radius_rsun"], "VIS")

def test_WASP189_apply_line_core_emission_IR():
    run_snapshot_apply_line_core_emission("WASP-189", "IR", SIGMA_MG22, SIGMA_MG21, STARS["WASP-189"]["log_r"], STARS["WASP-189"]["spectral_type"])

def test_WASP189_apply_line_core_emission_NUV():
    run_snapshot_apply_line_core_emission("WASP-189", "NUV", SIGMA_MG22, SIGMA_MG21, STARS["WASP-189"]["log_r"], STARS["WASP-189"]["spectral_type"])

def test_WASP189_apply_line_core_emission_VIS():
    run_snapshot_apply_line_core_emission("WASP-189", "VIS", SIGMA_MG22, SIGMA_MG21, STARS["WASP-189"]["log_r"], STARS["WASP-189"]["spectral_type"])

def test_WASP69_apply_line_core_emission_IR():
    run_snapshot_apply_line_core_emission("WASP-69", "IR", SIGMA_MG22, SIGMA_MG21, STARS["WASP-69"]["log_r"], STARS["WASP-69"]["spectral_type"])

def test_WASP69_apply_line_core_emission_NUV():
    run_snapshot_apply_line_core_emission("WASP-69", "NUV", SIGMA_MG22, SIGMA_MG21, STARS["WASP-69"]["log_r"], STARS["WASP-69"]["spectral_type"])

def test_WASP69_apply_line_core_emission_VIS():
    run_snapshot_apply_line_core_emission("WASP-69", "VIS", SIGMA_MG22, SIGMA_MG21, STARS["WASP-69"]["log_r"], STARS["WASP-69"]["spectral_type"])

def test_WASP69_apply_ism_absorption_IR():
    run_snapshot_apply_ism_absorption("WASP-69", "IR", STARS["WASP-69"]["EBV"])

def test_WASP69_apply_ism_absorption_NUV():
    run_snapshot_apply_ism_absorption("WASP-69", "NUV", STARS["WASP-69"]["EBV"])

def test_WASP69_apply_ism_absorption_VIS():
    run_snapshot_apply_ism_absorption("WASP-69", "VIS", STARS["WASP-69"]["EBV"])

def test_HD2685_apply_ism_absorption_IR():
    run_snapshot_apply_ism_absorption("HD 2685", "IR", STARS["HD 2685"]["EBV"])

def test_HD2685_apply_ism_absorption_NUV():
    run_snapshot_apply_ism_absorption("HD 2685", "NUV", STARS["HD 2685"]["EBV"])

def test_HD2685_apply_ism_absorption_VIS():
    run_snapshot_apply_ism_absorption("HD 2685", "VIS", STARS["HD 2685"]["EBV"])

def test_WASP69_compute_flux_at_earth_IR():
    run_snapshot_compute_flux_at_earth("WASP-69", "IR", STARS["WASP-69"]["distance_pc"])

def test_WASP69_compute_flux_at_earth_NUV():
    run_snapshot_compute_flux_at_earth("WASP-69", "NUV", STARS["WASP-69"]["distance_pc"])

def test_WASP69_compute_flux_at_earth_VIS():
    run_snapshot_compute_flux_at_earth("WASP-69", "VIS", STARS["WASP-69"]["distance_pc"])

def test_WASP189_compute_flux_at_earth_IR():
    run_snapshot_compute_flux_at_earth("WASP-189", "IR", STARS["WASP-189"]["distance_pc"])

def test_WASP189_compute_flux_at_earth_NUV():
    run_snapshot_compute_flux_at_earth("WASP-189", "NUV", STARS["WASP-189"]["distance_pc"])

def test_WASP189_compute_flux_at_earth_VIS():
    run_snapshot_compute_flux_at_earth("WASP-189", "VIS", STARS["WASP-189"]["distance_pc"])

def test_WASP69_apply_unred_IR():
    run_snapshot_apply_unred("WASP-69", "IR")

def test_WASP69_apply_unred_NUV():
    run_snapshot_apply_unred("WASP-69", "NUV")

def test_WASP69_apply_unred_VIS():
    run_snapshot_apply_unred("WASP-69", "VIS")

def test_HD2685_apply_unred_IR():
    run_snapshot_apply_unred("HD 2685", "IR")

def test_HD2685_apply_unred_NUV():
    run_snapshot_apply_unred("HD 2685", "NUV")

def test_HD2685_apply_unred_VIS():
    run_snapshot_apply_unred("HD 2685", "VIS")

def test_HD2685_convert_flux_to_photons_NUV():
    run_snapshot_convert_flux_to_photons("HD 2685", "NUV")

def test_HD2685_convert_flux_to_photons_VIS():
    run_snapshot_convert_flux_to_photons("HD 2685", "VIS")

def test_HD2685_convert_flux_to_photons_IR():
    run_snapshot_convert_flux_to_photons("HD 2685", "IR")

def test_WASP69_convert_flux_to_photons_NUV():
    run_snapshot_convert_flux_to_photons("WASP-69", "NUV")

def test_WASP69_convert_flux_to_photons_VIS():
    run_snapshot_convert_flux_to_photons("WASP-69", "VIS")

def test_WASP69_convert_flux_to_photons_IR():
    run_snapshot_convert_flux_to_photons("WASP-69", "IR")
