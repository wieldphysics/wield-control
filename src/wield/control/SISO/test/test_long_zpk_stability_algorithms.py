"""
"""
import numpy as np
import matplotlib.pyplot as plt
import control
from scipy.signal import cheby2
from wield.control import SISO
from wield.utilities.mpl import mplfigB

from wield.pytest import tjoin, fjoin, dprint  # noqa
from wield.control.utilities import algorithm_choice



def gen_filt():
    F_Fq_lo = 10 * 2 * np.pi
    F_Fq_hi = 120 * 2 * np.pi
    hp_order = 2 #lo_ord
    lp_order = 2
    F_p = []
    F_z = []
    for ord in range(hp_order):
        F_p.append(-F_Fq_lo + 0*1j*F_Fq_lo)
        F_z.append(-1)

    for ord in range(lp_order):
        F_p.append(-F_Fq_hi)

    F_k = 1
    c_zpk = cheby2(8, 160, 1.25*2*np.pi, btype='high', analog=True, output='zpk')
    F_z.extend(c_zpk[0])
    F_p.extend(c_zpk[1])
    F_k *= c_zpk[2]

    ian_lfq = -0.7
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

    filt = SISO.zpk(F_z, F_p, F_k, angular=True, fiducial_rtol=1e-5, fiducial_atol=1e-10)
    filtss = filt.asSS
    filt = filt / abs(filtss.Linf_norm()[0]) 
    return filt * filt


def gen_filt2():
    filt = SISO.zpk(
        [1e-2],
        [100e-2, 10e-2],
        10,
        angular=False,
        fiducial_rtol=1e-5,
        fiducial_atol=1e-10
    )
    filtss = filt.asSS
    # filt = filt / abs(filtss.Linf_norm()[0]) 
    return filt * filt * filt
    return filt



def test_long_ZPK_stability_plot():
    """
    This is a test of the numerical stability of a very large ZPK filter that spans 18 orders of magnitude

    """
    F_Hz = np.geomspace(3e-3, 1e7, 2000)
    axB = mplfigB(Nrows=2)

    filt = gen_filt()
    xfer = (filt).fresponse(f=F_Hz)
    # xfer = (filt * filt).fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer.fplot_mag, label="ZPK, wield")
    axB.ax1.semilogx(*xfer.fplot_deg180, label="ZPK, wield")

    filt_orig = filt
    zpk2ss_ranks = algorithm_choice.algorithm_choices_defaults['zpk2ss']
    dprint(zpk2ss_ranks)
    zpk2ss_names = list(zpk2ss_ranks.keys())
    # zpk2ss_names = ['zpk2ss_chain_poly']
    for k in zpk2ss_names:
        algorithm_choices = {'zpk2ss': {k: 1000}}
        filt = filt_orig.set_algorithm_choices(algorithm_choices)
        filtss = filt.asSS
        # filtss = filtss * filtss
        # print(filtss.A.shape, filtss.E.shape)

        filtss.print_nonzero()

        xfer = filtss.fresponse(f=F_Hz)
        label = "zpk2ss: {}".format(k)
        axB.ax0.loglog(*xfer.fplot_mag, label=label, alpha=0.5)
        axB.ax1.semilogx(*xfer.fplot_deg180, label=label)

    # axB.ax0.set_ylim(1e-25, 10)
    axB.ax0.axhline(np.finfo(float).eps, lw=1, ls='--', color='black')
    axB.ax0.axhline(filtss.ss.Nstates * np.finfo(float).eps, lw=1, ls='--', color='black')
    axB.ax0.axhline(filtss.ss.Nstates**2 * np.finfo(float).eps, lw=1, ls='--', color='black')
    axB.ax0.legend(fontsize=8)

    axB.save(tjoin('algorithm_compare_zpk2ss'))
    return


