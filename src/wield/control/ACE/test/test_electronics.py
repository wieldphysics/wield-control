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

from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.control.ACE import ACE
from wield.control.ACE import ace_electrical
from wield.control.algorithms.statespace.dense.xfer_algorithms import ss2xfer

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
    test_trigger,
    tpath_preclear,
)


c_m_s = 299792458


def print_ssd(ssd):
    print("Bnz", 1 * (ssd.B != 0))
    print("Anz", 1 * (ssd.A != 0))
    print("E", ssd.E)
    print("C", 1 * (ssd.C != 0))
    print("D", ssd.D)


def test_opamp(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    F_Hz = logspaced(0.01, 1e7, 100)

    ace = ACE.ACE()
    ace_op = ace_electrical.op_amp()
    ace.insert(ace_op, cmn="op1")
    # ace.insert(ace_electrical.voltage_source1(), cmn = 'Vp')
    # ace.insert(ace_electrical.voltage_source2(), cmn = 'Vn')
    # ace.bind_ports('op1.inP', 'Vp.')
    ace.bind_sum({"op1.outI"})
    ace.io_input("op1.posV")
    ace.io_input("op1.negV")
    ssB = ace.statespace(
        inputs=["op1.posV", "op1.negV"], outputs=["op1.outV"], Dreduce=False
    )
    printSSBnz(ssB)
    axB = mplfigB(Nrows=2)
    xfer = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=0)
    axB.ax0.loglog(F_Hz, abs(xfer))
    axB.ax1.semilogx(F_Hz, np.angle(xfer, deg=True))
    xfer = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=1)
    axB.ax0.loglog(F_Hz, abs(xfer))
    axB.ax1.semilogx(F_Hz, np.angle(xfer, deg=True))
    axB.ax0.axhline(1)
    axB.save(tpath_join("opamp_forward"))
    return


def test_opamp_fb(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    F_Hz = logspaced(0.01, 1e7, 100)

    ace = ACE.ACE()
    ace_op = ace_electrical.op_amp()
    ace.insert(ace_op, cmn="op1")
    ace.insert(ace_electrical.voltage_source1(), cmn="Vp")
    # ace.insert(ace_electrical.voltage_source2(), cmn = 'Vn')
    ace.bind_ports("op1.inP", "Vp.a")
    ace.io_input("Vp.V")
    ace.bind_ports("op1.out", "op1.inN")
    ssB = ace.statespace(inputs=["Vp.V"], outputs=["op1.outV"], Dreduce=False)
    dprint(ssB.A.shape)
    dprint(ssB.E.shape)
    printSSBnz(ssB)
    axB = mplfigB(Nrows=2)
    xfer = ss2xfer(*ssB.ABCDE, F_Hz=F_Hz, idx_in=0)
    axB.ax0.loglog(F_Hz, abs(xfer))
    axB.ax1.semilogx(F_Hz, np.angle(xfer, deg=True))
    axB.save(tpath_join("opamp_feedback"))
    return


def nz(M):
    return 1 * (M != 0)


def printSSBnz(ssb):
    Astr = np.array2string(nz(ssb.A))
    Bstr = np.array2string(nz(ssb.B))
    Cstr = np.array2string(nz(ssb.C))
    Dstr = np.array2string(nz(ssb.D))
    ziplines(Astr + "\n\n" + Cstr, Bstr + "\n\n" + Dstr, delim=" | ")


def ziplines(*args, delim=""):
    widths = []
    for arg in args:
        w = max(len(line) for line in arg.splitlines())
        widths.append(w)
    for al in zip(*[arg.splitlines() for arg in args]):
        line = []
        for a, w in zip(al, widths):
            line.append(a + " " * (w - len(a)))
        print(delim.join(line))
