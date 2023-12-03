"""
"""
import numpy as np
import matplotlib.pyplot as plt
import control
from scipy.signal import cheby2
from wield.control import SISO
from wield.utilities.mpl import mplfigB

from wield.pytest import tjoin, fjoin, dprint  # noqa

def gen_filt():
    F_Fq_lo = 10 * 2 * np.pi
    F_Fq_hi = 120 * 2 * np.pi
    hp_order = 1 #lo_ord
    lp_order = 2
    F_p = []
    F_z = []
    for ord in range(hp_order):
        F_p.append(-F_Fq_lo + 0*1j*F_Fq_lo)
        F_p.append(-F_Fq_lo - 0*1j*F_Fq_lo)
        F_z.append(0)
        F_z.append(0)

    for ord in range(lp_order):
        F_p.append(-F_Fq_hi)
    F_k = 1
    c_zpk = cheby2(8, 160, 1.25*2*np.pi, btype='high', analog=True, output='zpk')
    F_z.extend(c_zpk[0])
    F_p.extend(c_zpk[1])
    F_k *= c_zpk[2]

    ian_lfq = -0.7
    ian_lfq2 = -2
    ian_lfq3 = -0.5
    F_z.append(0)
    F_p.append(ian_lfq)
    F_z.append(0)
    F_p.append(ian_lfq)

    # F_z.append(ian_lfq3)
    # F_z.append(ian_lfq3)
    # F_p.append(ian_lfq2)
    # F_p.append(ian_lfq2)

    # make it more like a power spectrum
    F_p = F_p
    F_z = F_z
    # F_z = [0] *  13
    print()
    print("Poles list: ", "[" + "; ".join(["{} + {}j".format(p.real, p.imag) for p in F_p]), "]")
    print("Zeros list: ", "[" + "; ".join(["{} + {}j".format(z.real, z.imag) for z in F_z]), "]")

    filt = SISO.zpk(F_z, F_p, F_k, angular=True, fiducial_rtol=1e-5, fiducial_atol=1e-10)
    filtss = filt.asSS
    filt = filt / abs(filtss.Linf_norm()[0]) 
    return filt

def test_long_ZPK_stability():
    """
    This is a test of the numerical stability of a very large ZPK filter that spans 18 orders of magnitude

    """
    filt = gen_filt()
    # filt = filt * filt
    filtss = filt.asSS

    filt = filt / abs(filtss.Linf_norm()[0]) 
    filtss = filtss / abs(filtss.Linf_norm()[0]) 
    print("gain", filt.k)

    F_Hz = np.geomspace(1e-3, 1e3, 1000)

    axB = mplfigB(Nrows=2)

    mag, phase, omega = control.bode(control.ss(
        filtss.A,
        filtss.B,
        filtss.C,
        filtss.D,
    ), Hz=False, plot=False, omega=F_Hz*2*np.pi, wrap_phase=True, label = "python-control")
    axB.ax0.loglog(omega/(2*np.pi), mag, label="python-control")
    axB.ax1.semilogx(omega/(2*np.pi), phase * 180 / np.pi, label="python-control")

    xfer = filt.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="ZPK, wield")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="ZPK, wield")

    xfer = filtss.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="state space, wield")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="state space, wield")
    axB.ax0.set_ylim(1e-25, 10)
    axB.ax0.legend()

    axB.save(tjoin('long_zpk'))
    return


def test_long_cheby():
    """
    This is a demonstration showing the construction of a statespace using a
    Chechen companion matrix. 
    """
    filt = gen_filt()
    filt = filt
    p = filt.p
    z = filt.z
    norm = max(abs(p))**0.5
    norm = 1
    p = p / norm
    z = z / norm
    from numpy.polynomial.chebyshev import (
        chebcompanion,
        chebfromroots,
        chebroots,
        chebdiv,
    )
    cp = chebfromroots(p).real
    cz = chebfromroots(z).real
    nd = len(cp) - len(cz)
    if nd == 0:
        qz, cz = chebdiv(cz, cp)
        # surprisingly, the nd needs to stay 0 for the normalization call below,
        # rather than being reset
        # nd = len(cp) - len(cz)
    else:
        qz = 0
    print("nd: ", nd)
    czl = np.concatenate([cz, np.zeros(len(cp) - len(cz))])
    print("cp: ", cp)
    print("cz: ", cz)
    cp_alt = chebfromroots(p * 1j)
    print("cp alt!: ", cp_alt)
    A = chebcompanion(cp)
    Az = chebcompanion2(czl, scale = cp[-1])

    n = len(cp) - 1
    scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
    # this is basically the last line of the companion matrix, but not subtracting
    # the companion coupling alpha beta gamma contribution.
    czs = (czl[:-1]/cp[-1])*(scl/scl[-1])*.5

    print(A[:, -1])
    print(Az[:, -1])
    print("poles1: ", p * norm)
    print("poles1: ", chebroots(cp) * norm)
    p2 = chebroots(cp) * norm

    print("poles2: ", np.linalg.eigvals(A) * norm)
    z2 = z * norm

    filt2 = SISO.zpk(z2, p2, filt.k, angular=True, fiducial_rtol=1e-5, fiducial_atol=1e-10)

    C = np.zeros(len(cp) - 1)
    C[-1] = filt.k
    B = czs
    # B = np.zeros(len(cp) - 1)
    # B[0] = 1 / np.sqrt(.5)*(n-1)
    D = np.asarray([qz]).reshape(1, 1) * filt.k
    filt3 = SISO.statespace(
        A=A * norm,
        B=B.reshape(-1, 1) / norm**(nd - 1),
        C=C.reshape(1, -1),
        # C=B.reshape(1, -1),
        # B=C.reshape(-1, 1),
        D = D,
    )

    axB = mplfigB(Nrows=2)

    F_Hz = np.geomspace(1e-3, 1e5, 1000)

    #filt = filt * filt
    xfer = filt.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="ZPK, wield")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="ZPK, wield")
    axB.ax0.axhline(1e-16, ls='--', color='black', lw=1)

    #filt2 = filt2 * filt2
    xfer = filt2.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="filt2")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="filt2")
    #axB.ax0.set_ylim(1e-25, 10)
    axB.ax0.legend()

    #filt3 = filt3 * filt3
    xfer = filt3.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="filt3")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="filt3")
    axB.ax0.set_ylim(1e-25, 10)
    axB.ax0.legend()

    axB.save(tjoin('long_zpk'))
    return


def chebcompanion2(c, scale = None):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is a Chebyshev basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # c is a trimmed copy
    if scale is None:
        scale = c[-1]
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0]/c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[0] = np.sqrt(.5)
    top[1:] = 1/2
    bot[...] = top
    mat[:, -1] -= (c[:-1]/scale)*(scl/scl[-1])*.5
    return mat



def test_long_cascade():
    """
    This is a demonstration showing the construction of a statespace using a
    Chechen companion matrix. 
    """
    filt = SISO.zpk(
        [-3000] * 8,
        [-1] * 8,
        1,
        angular=True,
        fiducial_rtol=1e-5,
        fiducial_atol=1e-10
    )
    filtss = filt.asSS
    norm = abs(filtss.Linf_norm()[0])
    filt = filt.inv()
    filtss = filtss.inv()

    print(filtss.A)

    axB = mplfigB(Nrows=2)

    F_Hz = np.geomspace(1e-3, 1e5, 1000)

    xfer = filt.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="ZPK, wield")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="ZPK, wield")
    axB.ax0.axhline(1e-16, ls='--', color='black', lw=1)

    xfer = filtss.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="filt2")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="filt2")
    #axB.ax0.set_ylim(1e-25, 10)
    axB.ax0.legend()

    axB.save(tjoin('long_zpk'))
    return
