from utils.constants import AU_cm, Mgaratio_loggf2to1, MgII2w, MgII1w
import numpy as np
import logging
from utils.helpers import announce

def apply_line_core_emission(flux, sigmaMg22, sigmaMg21, logR, spectral_type, announce_user: bool = False):
    """
    Add Mg II h & k line core emission to the stellar flux.
    Thin wrapper around legacy cute_snr_lca.
    """
    announce("Starting to apply line core emission", announce_user)

    Rmg = compute_Rmg(spectral_type, logR)
    E=Rmg*AU_cm**2
    Mg21em=E/(1.0 + Mgaratio_loggf2to1)
    Mg22em=Mgaratio_loggf2to1*E/(1.+Mgaratio_loggf2to1)
    gaussMg22=gaussian(flux[:,0],MgII2w,sigmaMg22,0.3989*Mg22em/sigmaMg22)
    gaussMg21=gaussian(flux[:,0],MgII1w,sigmaMg21,0.3989*Mg21em/sigmaMg21)
    gaussMg2 = gaussMg21 + gaussMg22
    flux_emission = flux[:,1] + gaussMg2
    flux[:,1]= flux_emission

    logging.info("Line core emission applied: spectral_type=%s logR=%s sigmaMg22=%.6f sigmaMg21=%.6f Rmg=%s ", spectral_type, logR, sigmaMg22, sigmaMg21, Rmg)

    return flux

def compute_Rmg(stype, logR):
    """
    Compute Mg II line core emission scaling factor Rmg
    from spectral type and logR (log R'_HK).
    """
    if (stype == 'F5V' or stype == 'F6V' or stype == 'F7V' or 
        stype == 'F8V' or stype == 'F9V' or stype == 'F9.5V' or 
        stype == 'G0V' or stype == 'G1V' or stype == 'G2V' or 
        stype == 'G3V' or stype == 'G4V' or stype == 'G5V' or
        stype == 'G6V' or stype == 'G7V' or stype == 'G8V' or 
        stype == 'G9V') :
                c1 = 0.87
                c2 = 5.73
                Rmg = 10**(c1*logR+c2)
                regime = "F/G dwarf"

    elif (stype == 'K9V' or stype == 'K8V' or stype == 'K7V' or 
          stype == 'K6.5V' or  stype == 'K6V' or stype == 'K5.5V' or 
          stype == 'K5V' or stype == 'K4.5V' or stype == 'K4V' or 
          stype == 'K3.5V' or stype == 'K3V' or stype == 'K2.5V' or
          stype == 'K2V' or stype == 'K1.5V' or stype == 'K1V' or 
          stype == 'K0.5V' or stype == 'K0V') :
                c1 = 1.01
                c2 = 6.00
                Rmg = 10**(c1*logR+c2)
                regime = "K dwarf"

    elif (stype == 'M5V' or stype == 'M4V' or stype == 'M3V' or 
          stype == 'M2V' or stype == 'M2.5V' or stype == 'M1.5V' or 
          stype == 'M1V' or stype == 'M0.5V' or stype == 'M0V')  :
                c1 = 1.59
                c2 = 6.96
                Rmg = 10**(c1*logR+c2)
                regime = "M dwarf"

    else:
                Rmg = 0.0
                c1 = None
                c2 = None
                regime = "not applicable"

    logging.info("Rmg computed: stype=%s logR=%s regime=%s c1=%s c2=%s Rmg=%s", stype, logR, regime, c1, c2, Rmg)

    return Rmg


def gaussian(wavelength, wl0, sigma, scale=1.0):
  """
  Compute a Gaussian function centered at wl0, with width sigma, and
  height scale at wl0.
  """
  return scale * np.exp(-0.5*((wavelength-wl0)/sigma)**2)
