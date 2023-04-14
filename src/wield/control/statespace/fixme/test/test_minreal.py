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
import pytest
import copy
from wield import declarative
import wield.control

from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.control.algorithms.statespace import dense

import scipy.linalg
import IFO_model
import IFO_model_noM

pytestmark = pytest.mark.xfail(reason="Need to revisit these")


def test_controllable(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=4,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.rescale()
    # sys1.reduce(method = 'SVD')
    # sys1.permute_E_diagonal(location = 'upper left')
    # IFO_model.print_ssd(sys1)
    # sys1.controllable_staircase(reciprocal_system = True, do_reduce = True)
    # sys1.A = sys1.A[::-1, ::-1]
    # sys1.E = sys1.E[::-1, ::-1]
    # sys1.C = sys1.C[:, ::-1]
    # sys1.B = sys1.B[::-1, :]
    sys1.controllable_staircase(
        reciprocal_system=True,
        debug_path=tpath_join,
    )
    sys1.controllable_staircase(
        reciprocal_system=False,
        # debug_path = tpath_join,
    )

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


def test_controllable_octave(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=6,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()

    import oct2py

    oc = oct2py.Oct2Py()
    oc.eval("pkg load signal")
    oc.eval("pkg load control")

    oc.push("A", sys1.A)
    oc.push("B", sys1.B)
    oc.push("C", sys1.C)
    oc.push("D", sys1.D)
    oc.push("E", sys1.E)
    oc.eval("S=dss(A,B,C,D,E);")
    oc.eval("S=minreal(S);")
    oc.eval("A=S.A;B=S.B;C=S.C;D=S.D;E=S.E;")
    sys1.A = oc.pull("A")
    sys1.B = oc.pull("B")
    sys1.C = oc.pull("C")
    sys1.D = oc.pull("D")
    sys1.E = oc.pull("E")
    print(sys1.A)
    sys1.names_collect("states", to="states")
    sys1.names_collect("constr", to="constr")
    print(sys1c.A.shape, sys1.A.shape)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

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


def test_controllable_matlab(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=10,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()

    import matlab.engine
    import array

    ML = matlab.engine.start_matlab()

    def toArr(a):
        p = matlab.double(a.flatten("F").flatten().tolist())
        return ML.reshape(p, *a.shape)

    A = toArr(sys1.A)
    B = toArr(sys1.B)
    C = toArr(sys1.C)
    D = toArr(sys1.D)
    E = toArr(sys1.E)
    SS = ML.dss(A, B, C, D, E)
    SS = ML.minreal(SS, 1e-5)
    ML.workspace["SS"] = SS

    sys1.A = np.array(ML.eval("SS.A;"))
    sys1.B = np.array(ML.eval("SS.B;"))
    sys1.C = np.array(ML.eval("SS.C;"))
    sys1.D = np.array(ML.eval("SS.D;"))
    sys1.E = np.array(ML.eval("SS.E;"))
    if not sys1.E.shape or np.prod(sys1.E.shape) == 0:
        sys1.E = np.eye(sys1.A.shape[0])
    print(sys1.E.shape)
    sys1.names_collect("states", to="states")
    sys1.names_collect("constr", to="constr")
    print(sys1c.A.shape, sys1.A.shape)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

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


def test_controllable_slycotDSS(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=1,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.reduce(method="SVD")

    Ei = np.linalg.inv(sys1.E)
    sys1.A = Ei @ sys1.A
    sys1.E = Ei @ sys1.E
    sys1.B = Ei @ sys1.B

    import slycot

    print(sys1.A.shape)
    n = sys1.A.shape[0]
    m = sys1.B.shape[1]
    p = sys1.C.shape[0]
    mp = max(m, p)
    B = np.zeros((n, mp))
    B[:, :m] = sys1.B
    C = np.zeros((mp, n))
    C[:p, :] = sys1.C
    out = slycot._wrapper.tg01jd(
        "I",
        "R",
        "N",
        n,
        m,
        p,
        sys1.A,
        sys1.E,
        B,
        C,
        tol=1e-18,  # tol
    )
    A, E, B, C, NR, INFRED, INFO = out
    print(NR, INFRED, INFO)

    sys1.A = A  # np.copy(A[:NR, :NR])
    sys1.E = E  # np.copy(sys1.E[:NR, :NR])
    sys1.B = B  # np.copy(B[:NR, :])
    sys1.C = C  # np.copy(C[:, :NR])
    # print("A", sys1.A)
    # print("B", sys1.B)
    # print("C", sys1.C)
    # print("E", sys1.E)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

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


def test_controllable_slycotSS(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.00,
        space_order=1,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.reduce(method="SVD")

    Ei = np.linalg.inv(sys1.E)
    sys1.A = Ei @ sys1.A
    sys1.E = Ei @ sys1.E
    sys1.B = Ei @ sys1.B

    import slycot

    print(sys1.A.shape)
    n = sys1.A.shape[0]
    m = sys1.B.shape[1]
    p = sys1.C.shape[0]
    mp = max(m, p)
    B = np.zeros((n, mp))
    B[:, :m] = sys1.B
    C = np.zeros((mp, n))
    C[:p, :] = sys1.C
    out = slycot.transform.tb01pd(
        n,
        m,
        p,
        sys1.A,
        B,
        C,
        tol=1e-8,  # tol
        job="M",
    )
    A, B, C, NR = out
    print(NR)

    sys1.A = np.copy(A[:NR, :NR])
    sys1.E = np.copy(sys1.E[:NR, :NR])
    sys1.B = np.copy(B[:NR, :])
    sys1.C = np.copy(C[:, :NR])
    # print("A", sys1.A)
    # print("B", sys1.B)
    # print("C", sys1.C)
    # print("E", sys1.E)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

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


def test_controllable_slycot(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.00,
        space_order=1,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    sys1.reduce(method="SVD")

    import slycot

    print(sys1.A.shape)
    n = sys1.A.shape[0]
    m = sys1.B.shape[1]
    p = sys1.C.shape[0]
    mp = max(m, p)
    B = np.zeros((n, mp))
    B[:, :m] = sys1.B
    C = np.zeros((mp, n))
    C[:p, :] = sys1.C
    u, s, v = scipy.linalg.svd(sys1.E)
    print(s)
    out = slycot._wrapper.tg01jd(
        "I",
        "R",
        "N",
        n,
        m,
        p,
        sys1.A,
        sys1.E,
        B,
        C,
        0,  # tol
    )
    A, E, B, C, NR, INFRED, INFO = out
    print(NR, INFRED, INFO)

    sys1.A = np.copy(A[:NR, :NR])
    sys1.E = np.copy(E[:NR, :NR])
    sys1.B = np.copy(B[:NR, :])
    sys1.C = np.copy(C[:, :NR])
    # print("A", sys1.A)
    # print("B", sys1.B)
    # print("C", sys1.C)
    # print("E", sys1.E)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

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


def test_controllable_guptri(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.00,
        space_order=1,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1.reduce(method="diag")
    sys1.reduce(method="SVD")
    sys1c = sys1.copy()

    import guptri.guptri as guptri

    out = guptri.guptri(
        sys1.A,
        sys1.E,
        sys1.A.shape[0],
        sys1.A.shape[1],
        epsu=1e-12,
        gap=100,
    )

    (
        a,
        b,
        pp,
        qq,
        adelta,
        bdelta,
        rtre,
        rtce,
        zrre,
        zrce,
        fnre,
        fnce,
        inre,
        ince,
        pstru,
        stru,
        info,
    ) = out

    print(a)
    print(b)
    print(pstru)
    print(stru)
    print("Number right singular: ", rtre, rtce)
    print("Number Zero finite:    ", zrre - rtre, zrce - rtce)
    print("Number nonzero finite: ", fnre - zrre, fnce - zrce)
    print("number Infinite:       ", inre - fnre, ince - fnce)
    print("Number Left singular:  ", a.shape[0] - inre, a.shape[1] - ince)
    return


def test_controllable_slycot_xfer(test_trigger, tpath_join, plot):
    Zc = [-1 + 1j, -1 + 5j]
    Zr = [-100, -200]
    Pc = [-1 + 2j, -1 + 5j]
    Pr = [-10, -20, -30]
    k = 1
    sys1 = dense.ZPKdict(
        name="zpk",
        zdict=dict(c=Zc, r=Zr),
        pdict=dict(c=Pc, r=Pr),
        k=k,
    )
    sys1c = sys1.copy()

    import slycot

    print(sys1.A.shape)
    n = sys1.A.shape[0]
    m = sys1.B.shape[1]
    p = sys1.C.shape[0]
    mp = max(m, p)
    B = np.zeros((n, mp))
    B[:, :m] = sys1.B
    C = np.zeros((mp, n))
    C[:p, :] = sys1.C
    out = slycot._wrapper.tg01jd(
        "C",
        "R",
        "N",
        n,
        m,
        p,
        sys1.A,
        sys1.E,
        B,
        C,
        1e-16,  # tol
    )
    A, E, B, C, NR, INFRED, INFO = out
    print(NR, INFRED, INFO)

    # print(NR, INFRED, INFO)
    # out = slycot.tb01pd(
    #    sys1.A.shape[0],
    #    sys1.B.shape[1],
    #    sys1.C.shape[0],
    #    sys1.A,
    #    sys1.B,
    #    sys1.C,
    # )
    # print("REDUCED ORDER: ", out[-2])

    # out = slycot._wrapper.tg01hd(
    #    sys1.A.shape[0],
    #    sys1.B.shape[1],
    #    sys1.C.shape[0],
    #    sys1.A,
    #    sys1.E,
    #    sys1.B,
    #    sys1.C,
    #    1e-12,  # tol
    # )
    # if out[-1] < 0:
    #    error_text = "The following argument had an illegal value: "+arg_list[-out[-1]-1]
    #    raise ValueError(error_text)
    # A, E, B, C, Q, Z, NCONT, NIUCON, NRBLCK, TAU, INFO = out
    # print(NCONT, NIUCON, NRBLCK, TAU, INFO)
    sys1.A = np.copy(A[:NR, :NR])
    sys1.E = np.copy(E[:NR, :NR])
    sys1.B = np.copy(B[:NR, :])
    sys1.C = np.copy(C[:, :NR])
    print("A", sys1.A)
    print("B", sys1.B)
    print("C", sys1.C)
    print("E", sys1.E)
    sys1.constr.N = sys1.A.shape[0]
    sys1.constr.idx2N[-1] = sys1.A.shape[0]
    sys1.states.N = sys1.A.shape[1]
    sys1.states.idx2N[-1] = sys1.A.shape[1]

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        # iname = 'DARM.i0',
        # oname = 'Msrm+A-oP',
        oname="zpk.o0",
        iname="zpk.i0",
    )
    xfer_DARMc = sys1c.xfer(
        F_Hz=F_Hz,
        # iname = 'DARM.i0',
        # oname = 'Msrm+A-oP',
        oname="zpk.o0",
        iname="zpk.i0",
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


def test_controllable_guptri2(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=1,
        # no_QRPN = True,
    )
    Zc = [-1 + 1j, -1 + 5j]
    Zr = [-100, -200, -300]
    Pc = [-1 + 2j, -1 + 5j]
    Pr = [-10, -20, -30]
    k = 1
    sys1 = dense.ZPKdict(
        name="zpk",
        zdict=dict(c=Zc, r=Zr),
        pdict=dict(c=Pc, r=Pr),
        k=k,
    )
    sys1 = model.sys1
    sys1.reduce(method="diag")
    # sys1.reduce(method = 'SVD')
    sys1c = sys1.copy()

    import guptri.guptri as guptri
    import tabulate

    q, r = scipy.linalg.qr(sys1.B, mode="full")
    rank = r.shape[1]
    B = r
    A = q.T @ sys1.A
    E = q.T @ sys1.E
    AX = A[rank:, :].T
    EX = E[rank:, :].T
    print(tabulate.tabulate(r))
    print(tabulate.tabulate(q.T @ sys1.B))
    print(r.shape)
    print(sys1.A.shape)
    print(AX.shape)

    out = guptri.guptri(
        AX.astype(dtype=complex),
        EX.astype(dtype=complex),
        AX.shape[0],
        AX.shape[1],
        epsu=1e-6,
        gap=10,
        zero=1,
    )

    (
        a,
        b,
        pp,
        qq,
        adelta,
        bdelta,
        rtre,
        rtce,
        zrre,
        zrce,
        fnre,
        fnce,
        inre,
        ince,
        pstru,
        stru,
        info,
    ) = out
    print(adelta, bdelta)
    print(pp.shape, a.shape, qq.shape)
    print(a.shape, b.shape)
    Atry = pp.T.conjugate() @ AX @ qq
    print(tabulate.tabulate(a))
    print(tabulate.tabulate(Atry - a))

    print("Number right singular: ", rtre, rtce)
    print("Number Zero finite:    ", zrre - rtre, zrce - rtce)
    print("Number nonzero finite: ", fnre - zrre, fnce - zrce)
    print("number Infinite:       ", inre - fnre, ince - fnce)
    print("Number Left singular:  ", a.shape[0] - inre, a.shape[1] - ince)
    if False:
        N = 0
        a = a[N:, N:].T
        b = b[N:, N:].T
        qqT = pp.T
        ppT = qq.T

        A2 = np.block([[(A[:rank, :] @ qqT.T.conjugate())[:, N:]], [a]])
        E2 = np.block([[(E[:rank, :] @ qqT.T.conjugate())[:, N:]], [b]])
        B2 = B[: A2.shape[0], :]
        C2 = (sys1.C @ qqT)[:, N:]
        sys1.A = A2
        sys1.E = E2
        sys1.B = B2
        sys1.C = C2
        print(A2.shape, C2.shape)
    else:
        a = a.T
        b = b.T
        qqT = pp.T
        ppT = qq.T

        print(ppT.T.conjugate() @ ppT)
        print(qqT @ qqT.T.conjugate())
        A2 = np.block([[A[:rank, :] @ qqT.T.conjugate()], [AX.T @ qqT.T.conjugate()]])
        E2 = np.block([[E[:rank, :] @ qqT.T.conjugate()], [EX.T @ qqT.T.conjugate()]])
        B2 = B[: A2.shape[0], :]
        C2 = sys1.C @ qqT
        sys1.A = A2
        sys1.E = E2
        sys1.B = B2
        sys1.C = C2

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
        # oname = 'zpk.o0',
        # iname = 'zpk.i0',
    )
    xfer_DARMc = sys1c.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
        # oname = 'zpk.o0',
        # iname = 'zpk.i0',
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
