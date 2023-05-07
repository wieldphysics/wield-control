#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import numpy as np
import scipy
import scipy.signal

import pytest
from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
    test_trigger,
    tpath_preclear,
)


from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB

from wield.control import SISO
from wield.control.SISO import zpk_d2c_c2d

from scipy.signal import cheby2

# NOTE, should compute to normalize by the max of a filter
# using slycot.AB13DD, L-infinity norm of a state space system 

def FBNSsimpSS(gain=1):
    F_Fq_lo = 30 * 2 * np.pi
    F_Fq_hi = 300 * 2 * np.pi
    hp_order = 4 #lo_ord
    lp_order = 2
    F_p = []
    F_z = []
    for ord in range(1, hp_order + 1):
        F_p.append(-F_Fq_lo + 1j*F_Fq_lo)
        F_p.append(-F_Fq_lo - 1j*F_Fq_lo)
        F_z.append(0)
        F_z.append(0)
    for ord in range(1, lp_order + 1):
        F_p.append(-F_Fq_hi)
    F_k = 1

    filt = SISO.zpk((F_z, F_p, F_k), fiducial_f=[], fiducial_rtol=1, angular=False)
    filt = filt.normalize(f=200)
    return filt * gain


def FBNSsimpSSfull(gain=1):
    F_Fq_lo = 40 * 2 * np.pi
    F_Fq_hi = 300 * 2 * np.pi
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
    c_zpk = cheby2(8, 160, 4*2*np.pi, btype='high', analog=True, output='zpk')
    F_z.extend(c_zpk[0])
    F_p.extend(c_zpk[1])
    F_k *= c_zpk[2]

    filt = SISO.zpk((F_z, F_p, F_k), fiducial_f=[], fiducial_rtol=1, angular=False)
    filt = filt.normalize(f=200)
    return filt * gain


def test_ZPK_spectral_factorize(tpath_join):
    """
    """
    axB = mplfigB(Nrows=1)
    fs = 2048
    F_Hz = logspaced(1, 1000, 1000)

    ss_flat = SISO.zpk((), (), 1, fiducial_f=[]).asSS

    # test dynamic range with higher gain
    ss_bns = FBNSsimpSS(gain=100000).asSS

    ss_sq = ss_bns.conjugate() * ss_bns + ss_flat.conjugate() * ss_flat

    abssq = ss_flat.fresponse(f=F_Hz).mag**2 + ss_bns.fresponse(f=F_Hz).mag**2
    axB.ax0.loglog(F_Hz, abssq**0.5)

    fr = ss_sq.fresponse(f=F_Hz)
    axB.ax0.loglog(F_Hz, fr.mag**0.5)
    #axB.ax1.semilogx(*fr.fplot_deg225)

    # print(scipy.linalg.eigvals(ss_sq.A))

    ss_sqZPK = ss_sq.asZPK
    fr = ss_sqZPK.fresponse(f=F_Hz)
    axB.ax0.loglog(F_Hz, fr.mag**0.5)

    # from wield.control.algorithms.zpk import srootset, zrootset
    # classifier = srootset.default_root_classifier.classify_function(
    #     tRootSet=srootset.SDomainRootSet,
    #     hermitian=True,
    #     time_symm=True,
    #     return_unmatched=True
    # )

    def stable_root_extract(roots):
        """
        Return a square root filter

        TODO, make this numerically better behaved and t
        """
        roots = np.asarray(roots)
        lhp = roots[roots.real < 0]
        eq0 = roots[roots.real == 0]
        rhp = roots[roots.real > 0]
        assert(len(lhp) == len(rhp))
        assert(len(eq0) % 2 == 0)
        eq0 = sorted(eq0, key=lambda r: abs(r.imag))

        # skip every other real one
        return list(lhp) + list(eq0[::4]) + list(eq0[1::4])
    p = stable_root_extract(ss_sqZPK.p)
    z = stable_root_extract(ss_sqZPK.z)
    k = ss_sqZPK.k**0.5

    ss_rt = SISO.zpk(
        z,
        p,
        k,
        fiducial_f=[]
    )

    fr = ss_rt.fresponse(f=F_Hz)
    axB.ax0.loglog(*fr.fplot_mag)

    ss_rt2 = SISO.design.root_factored_quadrature_sum(ss_bns, ss_flat)
    fr = ss_rt.fresponse(f=F_Hz)
    axB.ax0.loglog(*fr.fplot_mag)

    #axB.ax1.semilogx(*fr.fplot_deg225)
    axB.save(tpath_join("Mag_show"))
    return 


def test_ZPK_spectral_factorize2(tpath_join):
    """
    """
    axB = mplfigB(Nrows=1)
    fs = 2048
    F_Hz = logspaced(1, 1000, 1000)

    ss_flat = SISO.zpk((), (), .001, fiducial_f=[]).asSS

    # test dynamic range with higher gain
    ss_bns = FBNSsimpSSfull(gain=100000).asSS

    ss_sq = ss_bns.conjugate() * ss_bns + ss_flat.conjugate() * ss_flat

    abssq = ss_flat.fresponse(f=F_Hz).mag**2 + ss_bns.fresponse(f=F_Hz).mag**2
    axB.ax0.loglog(F_Hz, abssq**0.5)

    fr = ss_sq.fresponse(f=F_Hz)
    axB.ax0.loglog(F_Hz, fr.mag**0.5, ls = "--")
    #axB.ax1.semilogx(*fr.fplot_deg225)

    # print(scipy.linalg.eigvals(ss_sq.A))

    ss_sqZPK = ss_sq.asZPK
    fr = ss_sqZPK.fresponse(f=F_Hz)
    axB.ax0.loglog(F_Hz, fr.mag**0.5, lw = 3, alpha = 0.5, ls=':')

    # from wield.control.algorithms.zpk import srootset, zrootset
    # classifier = srootset.default_root_classifier.classify_function(
    #     tRootSet=srootset.SDomainRootSet,
    #     hermitian=True,
    #     time_symm=True,
    #     return_unmatched=True
    # )

    def stable_root_extract(roots):
        """
        Return a square root filter

        TODO, make this numerically better behaved and t
        """
        roots = np.asarray(roots)
        lhp = roots[roots.real < 0]
        eq0 = roots[roots.real == 0]
        rhp = roots[roots.real > 0]
        assert(len(lhp) == len(rhp))
        assert(len(eq0) % 2 == 0)
        eq0 = sorted(eq0, key=lambda r: abs(r.imag))

        # skip every other real one
        return list(lhp) + list(eq0[::4]) + list(eq0[1::4])
    p = stable_root_extract(ss_sqZPK.p)
    z = stable_root_extract(ss_sqZPK.z)
    k = ss_sqZPK.k**0.5

    ss_rt = SISO.zpk(
        z,
        p,
        k,
        fiducial_f=[]
    )

    fr = ss_rt.fresponse(f=F_Hz)
    axB.ax0.loglog(*fr.fplot_mag, ls = ':')

    ss_rt2 = SISO.design.root_factored_quadrature_sum(ss_bns, ss_flat)
    fr = ss_rt.fresponse(f=F_Hz)
    axB.ax0.loglog(*fr.fplot_mag, ls = '--')

    #axB.ax1.semilogx(*fr.fplot_deg225)
    axB.save(tpath_join("Mag_show"))
    return 
