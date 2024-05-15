"""
Spectrum related functions

"""
import numpy as np
from toolbox4modeler import wave



def detrend_con(x):
    from statsmodels.tsa.tsatools import detrend
    
    return detrend(x, order=0)



def detrend_lin(x):
    from statsmodels.tsa.tsatools import detrend
    
    return detrend(x, order=1)



def detrend_quad(x):
    from statsmodels.tsa.tsatools import detrend
    
    return detrend(x, order=2)



def spectral_helper_function(f, Sp, h, IG_dvide=[1 / 26], rho=1023, g=9.81):
    # Inputs:
    # f         = frequency
    # Sp        = power spectral density of water surface elevation (m**2/Hz)
    # IG_dvide  = frequency spliting SS and IG
    # h         = water depth
    # rho       = water density

    # Outputs:

    # Repeat inputs
    # f         = frequency
    # Sp        = power spectral density of water surface elevation (m**2/Hz)

    # Outputs not binned
    # k         = wave number
    # L         = wave length
    # fp        = peak frequency
    # Tp        = peak period
    # Tmean     = mean period
    # fmean     = mean frequency
    # Tcent     = centriodal period
    # fcent     = centroidal frequency
    # SpTot     = intergated spectral density = m0

    # Outputs binned by frequency
    # E         = wave energy
    # F         = wave energy flux
    # Sxx       = radiation stress
    # Hrms      = rms wave height
    # Hsig      = sig wave height
    # Ustokes   = Stokes drift

    np.seterr(invalid="ignore")

    if not type(IG_dvide) == list:
        IG_dvide = [IG_dvide]

    if len(IG_dvide) == 1:
        freq_bins = np.array([[0, np.infty], [0, IG_dvide[0]], [IG_dvide[0], np.infty]])
        freq_bin_names = ["Total", "IG", "SS"]
    elif len(IG_dvide) == 2:
        freq_bins = np.array(
            [
                [0, np.infty],
                [0, min(IG_dvide[0])],
                [min(IG_dvide[0]), max(IG_dvide[0])],
                [max(IG_dvide[0]), np.infty],
            ]
        )

        # [0, min(IG_dvide), max(IG_dvide), np.infty])
        freq_bins_names = ["Total", "VL", "IG", "SS"]

    out = {}

    k = wave.disper(f / 2 / np.pi, h)
    L = (2 * np.pi) / k
    kd = k * h

    n = 1 / 2 + kd / np.sinh(2 * kd)  # Dean and Dalrymple pg 98
    # wave celerity
    C = np.sqrt(g / k * np.tanh(kd))  # Dean and Dalrymple pg 59
    # wave group velocity
    Cg = n * C
    Cg[1] = Cg[2]

    # Peak frequency
    fp = f[np.argmax(Sp)]
    # Peak period.
    Tp = 1 / fp
    # Mean period
    Tmean = np.trapz(f, Sp) / np.trapz(f, f * Sp)
    fmean = 1 / Tmean
    # Centriodal frequency
    fcent = sum(f * Sp) / sum(Sp)
    Tcent = 1 / fcent

    SpTot = {}
    E = {}
    F = {}
    Sxx = {}
    Hrms = {}
    Hsig = {}
    Ustokes = {}

    ii = -1
    for nn in freq_bin_names:
        # counter
        ii = ii + 1

        # frequency id
        id = (f > freq_bins[ii, 0]) & (f <= freq_bins[ii, 1])
        # intergated spectral density = m0
        SpTot[nn] = np.trapz(Sp[id], f[id])
        # wave energy
        E[nn] = np.trapz(rho * g * Sp[id], f[id])
        # wave energy flux
        F[nn] = np.trapz(rho * g * Cg[id] * Sp[id], f[id])
        # radiation stress, Dean and Dalrymple 2008 Eq 10.23
        Sxx[nn] = np.trapz(rho * g * (2 * n[id] - 0.5) * Sp[id], f[id])
        # rms wave height
        Hrms[nn] = np.sqrt(8 * SpTot[nn])
        # significant wave height
        Hsig[nn] = 4 * np.sqrt(SpTot[nn])

        # Predict stokes drift
        Ustokes[nn] = np.trapz(rho * g * Sp[id] / C[id] / rho / h, f[id])

    out = {}
    # storing inputs
    out["f"] = f
    out["Sp"] = Sp
    out["h"] = h

    # storing outputs
    out["Tp"] = Tp
    out["fp"] = fp
    out["Tmean"] = Tmean
    out["fmean"] = fmean
    out["Tcent"] = Tcent
    out["fcent"] = fcent
    out["L"] = L
    out["k"] = k

    out["SpTot"] = SpTot
    out["E"] = E
    out["F"] = F
    out["Sxx"] = Sxx
    out["Hrms"] = Hrms
    out["Hsig"] = Hsig
    out["Ustokes"] = Ustokes
    return out