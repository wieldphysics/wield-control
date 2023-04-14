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
import itertools

from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.control.SISO import design
from wield.control import SISO

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
    test_trigger,
    tpath_preclear,
)

import scipy.signal

c_m_s = 299792458



def test_Thiran_lqe(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ss = SISO.design.delay_thiran_raw(delay_s = 1e-3, order=6).asSS
    L, P = lqe(
        ss.A,
        ss.B,
        ss.C,
        1 * np.eye(ss.B.shape[1]),  # state noise, in this case driven by U via B
        1 * (ss.D.T @ ss.D + np.eye(ss.C.shape[0])), # observation noise (D contribution is from state noise driven by U)
    )
    A2 = ss.A - L @ ss.C
    dprint(ss.A)
    dprint(A2)
    ss2 = SISO.statespace(A=A2, B=ss.B, C=ss.C, D=ss.D)
    
    axB = mplfigB(Nrows=2)

    def trigger(fail, plot):
        F_Hz = logspaced(1, 1e4, 1000)
        xfer = ss.fresponse(f=F_Hz)
        xfer2 = ss2.fresponse(f=F_Hz)
        #(ss2 + ss)
        xfer3 = (ss2 * ss).fresponse(f=F_Hz)

        axB = mplfigB(Nrows=2)

        axB.ax0.loglog(*xfer.fplot_mag, label="Direct ZPK")
        axB.ax1.semilogx(*xfer.fplot_deg135, label="Direct ZPK")

        axB.ax0.loglog(*xfer2.fplot_mag, label="Direct ZPK")
        axB.ax1.semilogx(*xfer2.fplot_deg135, label="Direct ZPK")

        axB.ax0.loglog(*xfer3.fplot_mag, label="Direct ZPK")
        axB.ax1.semilogx(*xfer3.fplot_deg135, label="Direct ZPK")

        axB.save(tpath_join("test_final"))

    with test_trigger(trigger, plot=plot):
        pass
        # np.testing.assert_allclose(xfer, ABCD_xfer, rtol = 1e-5)
    return


def lqe(A, G, C, QN, RN):
    """
    NEEDS DOC
    """
    A = np.array(A, ndmin=2)
    G = np.array(G, ndmin=2)
    C = np.array(C, ndmin=2)
    QN = np.array(QN, ndmin=2)
    RN = np.array(RN, ndmin=2)
    import scipy.linalg

    P = scipy.linalg.solve_continuous_are(A.T, C.T, G @ QN @ G.T, RN)
    L = P @ C.T @ np.linalg.inv(RN)
    return L, P
