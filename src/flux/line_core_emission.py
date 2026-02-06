from domain.constants import AU, Mgaratio_loggf2to1, MgII2w, MgII1w
import numpy as np
import logging

def apply_line_core_emission(flux, sigmaMg22, sigmaMg21, logR, spectral_type):
    """
    Add Mg II h & k line core emission to the stellar flux.
    Thin wrapper around legacy cute_snr_lca.
    """
    print("Starting to apply line core emission")
    logging.info(
        "Applying Mg II line core emission: "
        "spectral_type=%s, logR=%s, sigmaMg22=%.6f, sigmaMg21=%.6f",
        spectral_type,
        logR,
        sigmaMg22,
        sigmaMg21,
    )
    flux_before = flux[:, 1].copy()

    Rmg = compute_Rmg(spectral_type, logR)
    E=Rmg*AU**2
    Mg21em=E/(1.0 + Mgaratio_loggf2to1)
    Mg22em=Mgaratio_loggf2to1*E/(1.+Mgaratio_loggf2to1)
    gaussMg22=gaussian(flux[:,0],MgII2w,sigmaMg22,0.3989*Mg22em/sigmaMg22)
    gaussMg21=gaussian(flux[:,0],MgII1w,sigmaMg21,0.3989*Mg21em/sigmaMg21)
    gaussMg2 = gaussMg21 + gaussMg22
    flux_emission = flux[:,1] + gaussMg2
    flux[:,1]= flux_emission

    # diff
    diff = flux[:, 1] - flux_before

    msg = (
        f"LCE DIFF max abs (full): {float(np.max(np.abs(diff)))} | "
        f"mean abs (full): {float(np.mean(np.abs(diff)))}"
    )
    print(msg)
    logging.info(msg)

    mg_mask = (flux[:, 0] >= 2790.0) & (flux[:, 0] <= 2850.0)
    if np.any(mg_mask):
        msg = (
            f"LCE DIFF max abs (Mg window): {float(np.max(np.abs(diff[mg_mask])))} | "
            f"mean abs (Mg window): {float(np.mean(np.abs(diff[mg_mask])))}"
        )
        print(msg)
        logging.info(msg)
    else:
        msg = "LCE DIFF Mg window: EMPTY"
        print(msg)
        logging.info(msg)

    return flux

def compute_Rmg(stype, logR):
    """
    Compute Mg II line core emission scaling factor Rmg
    from spectral type and logR (log R'_HK).
    """

    print("stype: ", stype)
    if (stype == 'F5V' or stype == 'F6V' or stype == 'F7V' or 
        stype == 'F8V' or stype == 'F9V' or stype == 'F9.5V' or 
        stype == 'G0V' or stype == 'G1V' or stype == 'G2V' or 
        stype == 'G3V' or stype == 'G4V' or stype == 'G5V' or
        stype == 'G6V' or stype == 'G7V' or stype == 'G8V' or 
        stype == 'G9V') :
                c1 = 0.87
                c2 = 5.73
                Rmg = 10**(c1*logR+c2)
    elif (stype == 'K9V' or stype == 'K8V' or stype == 'K7V' or 
          stype == 'K6.5V' or  stype == 'K6V' or stype == 'K5.5V' or 
          stype == 'K5V' or stype == 'K4.5V' or stype == 'K4V' or 
          stype == 'K3.5V' or stype == 'K3V' or stype == 'K2.5V' or
          stype == 'K2V' or stype == 'K1.5V' or stype == 'K1V' or 
          stype == 'K0.5V' or stype == 'K0V') :
                c1 = 1.01
                c2 = 6.00
                Rmg = 10**(c1*logR+c2)
    elif (stype == 'M5V' or stype == 'M4V' or stype == 'M3V' or 
          stype == 'M2V' or stype == 'M2.5V' or stype == 'M1.5V' or 
          stype == 'M1V' or stype == 'M0.5V' or stype == 'M0V')  :
                c1 = 1.59
                c2 = 6.96
                Rmg = 10**(c1*logR+c2)
    else:
                Rmg = 0.0
    return Rmg


def gaussian(wavelength, wl0, sigma, scale=1.0):
  """
  Compute a Gaussian function centered at wl0, with width sigma, and
  height scale at wl0.
  """
  return scale * np.exp(-0.5*((wavelength-wl0)/sigma)**2)
