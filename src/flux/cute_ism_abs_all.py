from utils.constants import C_LIGHT_km_s, ISM_b_Mg2, MgII1w, vr_ISM, MgII1_loggf, MgII1_stark, MgII2w, MgII2_loggf, MgII2_stark, MgIw, MgI_loggf, MgI_stark, FeIIw, FeII_loggf, FeII_stark
import scipy.constants as sc
import scipy.special as ss
import numpy as np

def cute_ism_abs_all(flux,n_mg2,n_mg1,n_fe2):
    #Construct the ISM absorption
    n_flux=flux[:,1]/flux[:,2]
    
    #for MgII doublet
    absorberMg1={'ion':'MG21','N': n_mg2,'B':ISM_b_Mg2,'Z': 0.0}
    lineMg1={'ion':'Mg21','wave':MgII1w+MgII1w*vr_ISM/C_LIGHT_km_s,'F':10**MgII1_loggf,'gamma':10**MgII1_stark}
    ISMMg21=voigtq(flux[:,0],absorberMg1,lineMg1)
    
    absorberMg2={'ion':'MG22','N':n_mg2,'B':ISM_b_Mg2,'Z':0.0}
    lineMg2={'ion':'Mg22','wave':MgII2w+MgII2w*vr_ISM/C_LIGHT_km_s,'F':10**MgII2_loggf,'gamma':10**MgII2_stark}
    ISMMg22=voigtq(flux[:,0],absorberMg2,lineMg2)
    
    #for MgI
    absorberMgI={'ion':'MG1','N':n_mg1,'B':ISM_b_Mg2,'Z':0.0}
    lineMgI={'ion':'Mg1','wave':MgIw+MgIw*vr_ISM/C_LIGHT_km_s,'F':10**MgI_loggf,'gamma':10**MgI_stark}
    ISMMg1=voigtq(flux[:,0],absorberMgI,lineMgI)
    
    #for FeII 
    absorberFeII={'ion':'FE2"','N':n_fe2,'B':ISM_b_Mg2,'Z':0.0}
    lineFeII={'ion':'Fe2','wave':FeIIw+FeIIw*vr_ISM/C_LIGHT_km_s,'F':10**FeII_loggf,'gamma':10**FeII_stark}
    ISMFe2=voigtq(flux[:,0],absorberFeII,lineFeII)

    #for CaII
    #absorberCaK=create_struct('ion','Ca2K','N',N,'B',ISM_b_Ca2,'Z',0.0)
    #lineCaK=create_struct('ion','Ca2K','wave',CaKwl0+CaKwl0*vr_ISM/vc,'F',10**CaK_loggf,'gamma',10**CaK_stark)
    #ISMCaK=voigtq(flux[0,*],absorberCaK,lineCaK)
    #
    #absorberCaH=create_struct('ion','Ca2H','N',N,'B',ISM_b_Ca2,'Z',0.0)
    #lineCaH=create_struct('ion','Ca2H','wave',CaHwl0+CaHwl0*vr_ISM/vc,'F',10**CaH_loggf,'gamma',10**CaH_stark)
    #ISMCaH=voigtq(flux[0,*],absorberCaH,lineCaH)
    
    ISM = ISMMg21*ISMMg22*ISMMg1*ISMFe2

    flux_absorption = ISM * n_flux
    flux[:,1]=flux[:,2]*flux_absorption

    return flux

def voigtq(wavelength, absorber, line):
    bnorm = absorber["B"] / C_LIGHT_km_s
    # Doppler width (Hz):
    vd = absorber["B"]*sc.kilo / (line["wave"] * sc.angstrom)

    vel = (wavelength/(line["wave"]*(1.0+absorber["Z"])) - 1.0)  / bnorm
    a = line["gamma"] / (4*np.pi * vd)

    idx_wings = np.abs(vel) >= 10.0
    idx_core = np.abs(vel) < 10.0

    vo = vel*0.0
    if np.any(idx_wings) > 0:
          vel2 = vel[idx_wings]**2
          hh1 = 0.56419/vel2 + 0.846/vel2**2
          hh3 = -0.56/vel2**2
          vo[idx_wings] = a * (hh1 + a**2 * hh3)

    if np.any(idx_core) > 0:
        x0 = 0.0
        hwhm_L = line["gamma"] / np.sqrt(2) / 2
        hwhm_G = vd * np.sqrt(np.log(2))
        voigt = Voigt(x0, hwhm_L, hwhm_G)
        vo[idx_core] = voigt(vel[idx_core]*vd)
        vo[idx_core] /= np.amax(vo[idx_core])

    tau = 0.014971475*(10.0**absorber["N"]) * line["F"] * vo/vd

    return np.exp(-tau)

class Voigt(object):
    """
    1D Voigt profile model.

    Parameters
    ----------
    x0: Float
       Line center location.
    hwhmL: Float
       Half-width at half maximum of the Lorentz distribution.
    hwhmG: Float
       Half-width at half maximum of the Gaussian distribution.
    scale: Float
       Scale of the profile (scale=1 returns a profile with integral=1.0).

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyratbay.opacity.broadening as b
    >>> Nl = 5
    >>> Nw = 10.0
    >>> hG = 1.0
    >>> HL = np.logspace(-2, 2, Nl)
    >>> l = b.Lorentz(x0=0.0)
    >>> d = b.Gauss  (x0=0.0, hwhm=hG)
    >>> v = b.Voigt  (x0=0.0, hwhmG=hG)

    >>> plt.figure(11, (6,6))
    >>> plt.clf()
    >>> plt.subplots_adjust(0.15, 0.1, 0.95, 0.95, wspace=0, hspace=0)
    >>> for i in np.arange(Nl):
    >>>   hL = HL[i]
    >>>   ax = plt.subplot(Nl, 1, 1+i)
    >>>   v.hwhmL = hL
    >>>   l.hwhm  = hL
    >>>   width = 0.5346*hL + np.sqrt(0.2166*hL**2+hG**2)
    >>>   x = np.arange(-Nw*width, Nw*width, width/1000.0)
    >>>   plt.plot(x/width, l(x), lw=2.0, color="b",         label="Lorentz")
    >>>   plt.plot(x/width, d(x), lw=2.0, color="limegreen", label="Doppler")
    >>>   plt.plot(x/width, v(x), lw=2.0, color="orange",    label="Voigt",
    >>>            dashes=(8,2))
    >>>   plt.ylim(np.amin([l(x), v(x)]), 3*np.amax([l(x), v(x), d(x)]))
    >>>   ax.set_yscale("log")
    >>>   plt.text(0.025, 0.75, r"$\rm HW_L/HW_G={:4g}$".format(hL/hG),
    >>>            transform=ax.transAxes)
    >>>   plt.xlim(-Nw, Nw)
    >>>   plt.xlabel(r"$\rm x/HW_V$", fontsize=12)
    >>>   plt.ylabel(r"$\rm Profile$")
    >>>   if i != Nl-1:
    >>>       ax.set_xticklabels([""])
    >>>   if i == 0:
    >>>       plt.legend(loc="upper right", fontsize=11)
    """
    def __init__(self, x0=0.0, hwhmL=1.0, hwhmG=1.0, scale=1.0):
        # Profile parameters:
        self.x0    = x0
        self.hwhmL = hwhmL
        self.hwhmG = hwhmG
        self.scale = scale
        # Constants:
        self._A = np.array([-1.2150, -1.3509, -1.2150, -1.3509])
        self._B = np.array([ 1.2359,  0.3786, -1.2359, -0.3786])
        self._C = np.array([-0.3085,  0.5906, -0.3085,  0.5906])
        self._D = np.array([ 0.0210, -1.1858, -0.0210,  1.1858])
        self._sqrtln2 = np.sqrt(np.log(2.0))
        self._sqrtpi  = np.sqrt(np.pi)


    def __call__(self, x):
        return self.eval(x)


    def eval(self, x):
        """
        Compute Voigt profile over the specified coordinates range.

        Parameters
        ----------
        x: 1D float ndarray
           Input coordinates where to evaluate the profile.

        Returns
        -------
        v: 1D float ndarray
           The line profile at the x locations.
        """
        if self.hwhmL/self.hwhmG < 0.1:
            sigma = self.hwhmG / (self._sqrtln2 * np.sqrt(2))
            z = (x + 1j * self.hwhmL - self.x0) / (sigma * np.sqrt(2))
            return self.scale * ss.wofz(z).real / (sigma * np.sqrt(2*np.pi))

        # This is faster than the previous script (but fails for HWl/HWg > 1.0):
        X = (x-self.x0) * self._sqrtln2 / self.hwhmG
        Y = self.hwhmL * self._sqrtln2 / self.hwhmG

        V = 0.0
        for i in np.arange(4):
            V += (self._C[i]*(Y-self._A[i]) + self._D[i]*(X-self._B[i])) \
                 / ((Y-self._A[i])**2 + (X-self._B[i])**2)
        V /= np.pi * self.hwhmL
        return \
            self.scale * self.hwhmL/self.hwhmG * self._sqrtpi*self._sqrtln2 * V
