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
import copy
from wield import declarative
import wield.control
import pytest

from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.control.algorithms.statespace import dense
from wield.control.algorithms.statespace.dense import reduce_algorithms

import IFO_model


pytestmark = pytest.mark.xfail(reason="Need to revisit these")


def test_slycotSS(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=6,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.rescale(method="slycot")
    sys1.reduce(method="SVD")

    (A, B, C, D, E), Ndiag, NcoBlk, NstBlk = reduce_algorithms.permute_Ediag_inplace(
        sys1.ABCDE,
    )
    assert Ndiag == NcoBlk and Ndiag == NstBlk
    assert np.all(E[:Ndiag, :Ndiag] == np.eye(Ndiag))
    print("NDiagonal, ", Ndiag, " of ", A.shape[0])

    import slycot

    print(sys1.A.shape)
    if True:
        n = sys1.A.shape[0]
        m = sys1.B.shape[1]
        p = sys1.C.shape[0]
        out = slycot.transform.tb01id(
            n,
            m,
            p,
            3,
            A,
            B,
            C,
            job="A",
        )
        s_norm, A, B, C, scale = out
        print("SCALE", scale)
        print("s_norm", s_norm)

    n = Ndiag
    B2 = np.block([[B[:Ndiag, :], A[:Ndiag, Ndiag:]]])
    m = B2.shape[1]
    out = slycot.analysis.ab01nd(
        n,
        m,
        np.copy(A[:Ndiag, :Ndiag]),
        np.copy(B2[:Ndiag, :]),
        tol=1e-7,  # tol
        jobz="I",
    )
    Ac, Bc, Ncont, indcon, nblk, Z, tau = out
    print("UNCONTROLLABLE", n - Ncont)
    if False:
        A[:Ndiag, :Ndiag] = Z.T @ A[:Ndiag, :Ndiag]
        A[:Ndiag, Ndiag:] = Bc[:Ndiag, B.shape[1] :]
    else:
        A[:Ndiag, :] = Z.T @ A[:Ndiag, :]
    A[:, :Ndiag] = A[:, :Ndiag] @ Z
    B[:Ndiag, :] = Bc[:Ndiag, : B.shape[1]]
    C[:, :Ndiag] = C[:, :Ndiag] @ Z

    import tabulate

    print("Astar", tabulate.tabulate(A[Ncont:Ndiag, Ndiag:]))
    #### STAGE 2

    # B3 = A[Ndiag:, Ncont:Ndiag].T
    # A3 = A[Ncont:Ndiag, Ncont:Ndiag].T
    # n = A3.shape[0]
    # m = B3.shape[1]
    # out = slycot.analysis.ab01nd(
    #    n,
    #    m,
    #    np.copy(A3),
    #    np.copy(B3),
    #    tol = 1e-6,  # tol
    #    jobz = 'I',
    # )
    # Ac2, Bc2, Ncont2, indcon2, nblk2, Z2, tau2 = out
    # print("UNCONTROLLABLE", n - Ncont2)
    # A[Ncont:Ndiag, :] = Z2 @ A[Ncont:Ndiag, :]
    # A[:, Ncont:Ndiag] = A[:, Ncont:Ndiag] @ Z2.T
    ##B[:Ndiag, :] = Bc[:Ndiag, :B.shape[1]]
    # C[:, Ncont:Ndiag] = C[:, Ncont:Ndiag] @ Z2.T

    # n = A.shape[0]
    # m = B.shape[1]
    # p = C.shape[0]
    # mp = max(m, p)
    # B2 = np.zeros((n, mp))
    # B2[:, :m] = B
    # C2 = np.zeros((mp, n))
    # C2[:p, :] = C
    # out = slycot.transform.tb01pd(
    #    n,
    #    m,
    #    p,
    #    A,
    #    B2,
    #    C2,
    #    tol = 1e-8,  # tol
    #    job = 'C',
    #    equil = 'N',
    # )
    # A, B, C, NR = out
    # print(NR)

    # sys1.A = np.copy(A[:NR, :NR])
    # sys1.E = np.copy(sys1.E[:NR, :NR])
    # sys1.B = np.copy(B[:NR, :])
    # sys1.C = np.copy(C[:, :NR])

    # import tabulate
    # print(tabulate.tabulate(Z))

    sys1.A = A
    sys1.B = B
    sys1.C = C
    sys1.D = D
    sys1.E = E

    # print("A", sys1.A)
    # print("B", sys1.B)
    # print("C", sys1.C)
    # print("E", sys1.E)

    # sys1.constr.N = sys1.A.shape[0]
    # sys1.constr.idx2N[-1] = sys1.A.shape[0]
    # sys1.states.N = sys1.A.shape[1]
    # sys1.states.idx2N[-1] = sys1.A.shape[1]

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )
    xfer_DARMc = sys1c.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )
    print()
    # IFO_model.print_ssd(sys1)

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_DARM))
        axB.ax0.loglog(F_Hz, abs(xfer_DARMc))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARMc, deg=True))
        axB.save(tpath_join("DARM"))

    with test_trigger(trigger, plot=plot):
        np.testing.assert_almost_equal(xfer_DARM / xfer_DARMc, 1, decimal=5)
    return


def test_rescale_slycot(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=4,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.rescale(method="slycot")

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )
    xfer_DARMc = sys1c.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )
    print()
    IFO_model.print_ssd(sys1)

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_DARM))
        axB.ax0.loglog(F_Hz, abs(xfer_DARMc))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARMc, deg=True))
        axB.save(tpath_join("DARM"))

    with test_trigger(trigger, plot=plot):
        np.testing.assert_almost_equal(xfer_DARM / xfer_DARMc, 1, decimal=5)
    return
