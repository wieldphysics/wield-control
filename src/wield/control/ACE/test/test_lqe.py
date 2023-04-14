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
from wield.control.algorithms.statespace import dense
from wield.control.ACE import ACE
from wield.control.algorithms.statespace.dense.xfer_algorithms import ss2xfer

from wield.AAA import AAA

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


def print_ssd(ssd):
    print("Bnz", 1 * (ssd.B != 0))
    print("Anz", 1 * (ssd.A != 0))
    print("E", ssd.E)
    print("C", 1 * (ssd.C != 0))
    print("D", ssd.D)


def test_lqe1(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()

    seis_acc = dense.zpk_rc(
        name="seis_acc",
        Zc=[],
        Zr=[-30],
        Pc=[-0.10 - 0.200j],
        Pr=[
            -10,
        ],
        k=1e-6,
        convention="scipyHz",
    )
    Aseis_acc = ACE.ACE.from_ABCD(*seis_acc.ABCDE)
    ace.insert(Aseis_acc, cmn="g1")

    seis1_sensing = dense.zpk_rc(
        name="seis1",
        Zc=[],
        Zr=[-0.900, -0.900],
        Pc=[],
        Pr=[-0.001, -0.001],
        k=1e-9,
        convention="scipyHz",
    )
    Aseis1_sensing = ACE.ACE.from_ABCD(*seis1_sensing.ABCDE)
    ace.insert(Aseis1_sensing, cmn="s1")
    seis2_sensing = dense.zpk_rc(
        name="seis2",
        Zc=[],
        Zr=[],
        Pc=[],
        Pr=[],
        k=1e-7,
        convention="scipyHz",
    )
    Aseis2_sensing = ACE.ACE.from_ABCD(*seis2_sensing.ABCDE)
    ace.insert(Aseis2_sensing, cmn="s2")

    weights_acc = dense.zpk_rc(
        name="weights_acc",
        Zr=[],
        # Zc = [-1 - 1j],
        # Pr = [-10, -10],
        Zc=[-0.1 - 0.1j],
        Pr=[-1, -1],
        Pc=[],
        k=1,
        convention="scipyHz",
    )
    Aweights_acc = ACE.ACE.from_ABCD(*weights_acc.ABCDE)

    F_Hz = logspaced(0.001, 1e3, 1000)

    g1_xfer = seis_acc.xfer(F_Hz, iname="zpk.i0", oname="zpk.o0")
    s1_xfer = seis1_sensing.xfer(F_Hz, iname="zpk.i0", oname="zpk.o0")
    s2_xfer = seis2_sensing.xfer(F_Hz, iname="zpk.i0", oname="zpk.o0")
    xfer_opt = (+abs(g1_xfer) ** -2 + abs(s1_xfer) ** -2 + abs(s2_xfer) ** -2) ** -0.5
    wTF = AAA.tfAAA(F_Hz, 1 / (1e13 * xfer_opt ** 2), degree_max=11)

    weights_acc2 = dense.zpk_rc(
        name="weights_acc",
        Zr=[r.real for r in wTF.zeros if r.imag == 0 and r.real < 0],
        Zc=[r for r in wTF.zeros if r.imag > 0 and r.real < 0],
        Pr=[r.real for r in wTF.poles if r.imag == 0 and r.real < 0] + [-100, -100],
        Pc=[r for r in wTF.poles if r.imag > 0 and r.real < 0],
        k=abs(wTF.gain) ** -0.5 * 10000,
        convention="scipyHz",
    )
    Aweights_acc2 = ACE.ACE.from_ABCD(*weights_acc2.ABCDE)
    # determines if the "optimal AAA weighting should be used"
    if True:
        weights_acc = weights_acc2
        Aweights_acc = Aweights_acc2

    ace.insert(Aweights_acc, cmn="wg1")
    ace.insert(Aweights_acc, cmn="wO1")
    ace.insert(Aweights_acc, cmn="ws1")
    ace.insert(Aweights_acc, cmn="ws2")

    dprint([(d["order"], d["res_rms"]) for d in wTF.fit_list])
    dprint(wTF.supports)
    dprint(wTF.zpk)
    axB = mplfigB()

    w_xfer = weights_acc.xfer(F_Hz, iname="zpk.i0", oname="zpk.o0")
    w_xfer2 = weights_acc2.xfer(F_Hz, iname="zpk.i0", oname="zpk.o0")
    axB.ax0.loglog(F_Hz, abs(g1_xfer), lw=2, label="Seismic Background")
    ax0B = axB.ax0.twinx()
    ax0B.loglog(
        F_Hz, abs(w_xfer), color="black", ls="--", lw=1, label="weighting function"
    )
    ax0B.loglog(
        F_Hz, abs(w_xfer2), color="purple", ls="--", lw=1, label="weighting function"
    )
    axB.ax0.loglog(
        F_Hz, abs(s1_xfer), label="Sensor 1, good but tilt-contaminated", lw=2
    )
    ax0B.set_ylabel("Weighting Magnitude [black dashed]")
    axB.ax0.loglog(F_Hz, abs(s2_xfer), label="Sensor 2, bad but broadband", lw=2)
    axB.ax0.set_xlabel("Frequency [Hz]")
    axB.ax0.set_ylabel("Seismic Acceleration-like units")
    axB.ax0.loglog(
        F_Hz, abs(xfer_opt), color="magenta", lw=2, label="aspirational optimum"
    )

    axB.ax0.legend(loc="lower left", framealpha=1)
    ax0B.legend(loc="lower right", framealpha=1)
    # Re-arrange legends to last axis
    all_axes = axB.fig.get_axes()
    for axis in all_axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
            all_axes[-1].add_artist(legend)

    axB.save(tpath_join("inputs"))

    ace.bind_equal({"g1.O", "s2.O"})
    ace.bind_equal({"g1.O", "s1.O"})
    ace.bind_equal({"ws1.O", "s2.I"})
    ace.bind_equal({"ws2.O", "s1.I"})
    ace.bind_equal({"wg1.O", "g1.I"})
    ace.bind_equal({"wO1.O", "g1.O"})
    ace.io_input("wg1.I")

    ssB = ace.statespace(
        inputs=["wg1.I"], outputs=["ws1.I", "ws2.I", "g1.I"], Dreduce=False
    )
    xfer1 = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_out=0)
    ACE.printSSBnz(ssB)

    sccs = ace.states_reducible_sccs()
    dprint("scc2", sccs)
    for st_set, cr_set in sccs[:]:
        dprint("SCC_reduce", st_set, cr_set)
        ace.simplify_scc(st_set, cr_set)

    ssB = ace.statespace(
        inputs=["wg1.I"], outputs=["ws1.I", "ws2.I", "g1.O"], Dreduce=True
    )
    ACE.printSSBnz(ssB)

    xfer2A = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_out=0)
    xfer2B = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_out=1)

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer1))
        axB.ax1.semilogx(F_Hz, np.angle(xfer1, deg=True))
        axB.ax0.loglog(F_Hz, abs(xfer2A))
        axB.ax1.semilogx(F_Hz, np.angle(xfer2A, deg=True))
        axB.ax0.loglog(F_Hz, abs(xfer2B))
        axB.ax1.semilogx(F_Hz, np.angle(xfer2B, deg=True))
        axB.save(tpath_join("test_ssB0"))

    with test_trigger(trigger, plot=plot):
        pass
        # np.testing.assert_allclose(xfer, ABCD_xfer, rtol = 1e-5)

    import control

    dprint("D", ssB.D)
    A = ssB.A
    B = ssB.B
    C = ssB.C[:2, :]
    D = ssB.D[:2, :]
    # L, P, E = control.lqe(
    L, P = lqe(
        A,
        B,
        C,
        np.eye(B.shape[1]),
        D.T @ D + np.eye(C.shape[0]),
        # D,
    )
    A2 = ssB.A - L @ C

    ABCD = A2, L, ssB.C[2:3, :], np.zeros((1, 2))
    xfer00 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=0, idx_out=0)
    xfer10 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=1, idx_out=0)
    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(xfer00))
    axB.ax1.semilogx(F_Hz, np.angle(xfer00, deg=True))
    axB.ax0.loglog(F_Hz, abs(xfer10))
    axB.ax1.semilogx(F_Hz, np.angle(xfer10, deg=True))
    axB.save(tpath_join("test_lqe"))

    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(xfer00 / w_xfer))
    axB.ax1.semilogx(F_Hz, np.angle(xfer00 / w_xfer, deg=True))
    axB.ax0.loglog(F_Hz, abs(xfer10 / w_xfer))
    axB.ax1.semilogx(F_Hz, np.angle(xfer10 / w_xfer, deg=True))
    axB.save(tpath_join("test_lqeW"))

    ace2 = ACE.ACE()
    ace2.insert(ACE.ACE.from_ABCD(*ssB.ABCDE), cmn="env")
    ace2.insert(ACE.ACE.from_ABCD(*ABCD), cmn="kal")
    ace2.insert(Aweights_acc, cmn="wO1")

    ace2.io_add("env.O1", matmap={"env.O": [0]})
    ace2.io_add("env.O2", matmap={"env.O": [1]})
    ace2.io_add("env.g", matmap={"env.O": [2]})
    ace2.io_add("kal.I1", matmap={"kal.I": [0]})
    ace2.io_add("kal.I2", matmap={"kal.I": [1]})
    ace2.io_add("kal.g", matmap={"kal.O": [0]})

    ace2.states_augment(N=1, st="kal.v1", io=True)
    ace2.states_augment(N=1, st="kal.v2", io=True)
    ace2.io_input("kal.v1")
    ace2.io_input("kal.v2")
    ace2.io_input("env.I")

    ace2.bind_sum({"env.O1": 1, "kal.v1": 1, "kal.I1": -1})
    ace2.bind_sum({"env.O2": 1, "kal.v2": 1, "kal.I2": -1})
    ace2.bind_sum({"kal.g": 1, "env.g": -1, "wO1.O": 1})

    ssB = ace2.statespace(inputs=["env.I", "kal.v1", "kal.v2"], outputs=["wO1.I"])
    ACE.printSSBnz(ssB)

    xfer_g1 = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=0, idx_out=0)
    xfer_s1 = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=1, idx_out=0)
    xfer_s2 = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=2, idx_out=0)

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_g1), label="Seismic Background", lw=2)
        axB.ax1.semilogx(
            F_Hz, np.angle(xfer_g1, deg=True), label="Seismic Background", lw=2
        )
        axB.ax0.loglog(
            F_Hz, abs(xfer_s1), label="Sensor 1, good but tilt-contaminated", lw=2
        )
        axB.ax1.semilogx(
            F_Hz,
            np.angle(xfer_s1, deg=True),
            label="Sensor 1, good but tilt-contaminated",
            lw=2,
        )
        axB.ax0.loglog(F_Hz, abs(xfer_s2), label="Sensor 2, bad but broadband", lw=2)
        axB.ax1.semilogx(
            F_Hz, np.angle(xfer_s2, deg=True), label="Sensor 2, bad but broadband", lw=2
        )
        axB.ax0.loglog(
            F_Hz, abs(xfer_opt), color="magenta", lw=2, label="aspirational optimum"
        )
        axB.ax0.set_xlabel("Frequency [Hz]")
        axB.ax0.set_ylabel("Seismic Acceleration-like units")
        axB.ax0.legend()
        axB.ax0.set_ylim(1e-14, 1e-6)
        axB.save(tpath_join("test_final"))

    with test_trigger(trigger, plot=plot):
        pass
        # np.testing.assert_allclose(xfer, ABCD_xfer, rtol = 1e-5)
    return


def test_lqe2(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    A = np.array(
        [
            [0.00000000e00, -5.00000000e-01, 0.00000000e00, 0.00000000e00],
            [3.94784176e00, -1.25663706e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 9.99900000e01, 0.00000000e00, -1.00000000e02],
            [0.00000000e00, 1.24407069e00, 3.94784176e-03, -1.25663706e00],
        ]
    )
    B = np.array(
        [[1.50000000e-06], [3.76991118e-06], [9.99900000e-05], [1.24407069e-06]]
    )
    C = np.array([[0.0e00, 1.0e09, 0.0e00, -1.0e09], [0.0e00, 1.0e07, 0.0e00, 0.0e00]])
    D = np.array([[1000.0], [10.0]])
    import control

    L, P, E = control.lqe(
        A,
        B,
        C,
        np.eye(B.shape[1]),
        D.T @ D + np.eye(C.shape[0]),
        D,
    )
    A2 = A - L @ C

    F_Hz = logspaced(0.001, 1e2, 1000)

    ABCD = A2, L, C, np.zeros((2, 2))
    xfer00 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=0, idx_out=0)
    xfer10 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=1, idx_out=0)
    xfer01 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=0, idx_out=1)
    xfer11 = ss2xfer(*ABCD, F_Hz=F_Hz, idx_in=1, idx_out=1)
    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(xfer00))
    axB.ax1.semilogx(F_Hz, np.angle(xfer00, deg=True))
    axB.ax0.loglog(F_Hz, abs(xfer01))
    axB.ax1.semilogx(F_Hz, np.angle(xfer01, deg=True))
    axB.ax0.loglog(F_Hz, abs(xfer01))
    axB.ax1.semilogx(F_Hz, np.angle(xfer10, deg=True))
    axB.ax0.loglog(F_Hz, abs(xfer10))
    axB.ax1.semilogx(F_Hz, np.angle(xfer11, deg=True))

    axB.save(tpath_join("test_ssB0"))


def lqe(A, G, C, QN, RN):
    A, G, C = np.array(A, ndmin=2), np.array(G, ndmin=2), np.array(C, ndmin=2)
    QN, RN = np.array(QN, ndmin=2), np.array(RN, ndmin=2)
    import scipy.linalg

    P = scipy.linalg.solve_continuous_are(A.T, C.T, np.dot(np.dot(G, QN), G.T), RN)
    L = P @ C.T @ np.linalg.inv(RN)
    return L, P
