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
from wield.control.statespace import dense, StateSpaceDense

import IFO_model
import IFO_model_noM

from wield.pytest.fixtures import (
    tpath_join,
    dprint,
)


# TODO, should check this against the analytical model in setups
def test_IFO_model(tpath_join):
    model = IFO_model.build_model(theta=0.01)
    sys1 = model.sys1

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )

    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(xfer_DARM))
    axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
    axB.save(tpath_join("DARM"))

    xfer_REFLPQ = sys1.xfer(
        F_Hz=F_Hz,
        iname="SRMQ.i0",
        oname="Msrm+A-oP",
    )

    xfer_REFLPP = sys1.xfer(
        F_Hz=F_Hz,
        iname="SRMP.i0",
        oname="Msrm+A-oP",
    )

    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(xfer_REFLPP))
    axB.ax0.loglog(F_Hz, abs(xfer_REFLPQ))
    axB.ax1.semilogx(F_Hz, np.angle(xfer_REFLPP, deg=True))
    axB.ax1.semilogx(F_Hz, np.angle(xfer_REFLPQ, deg=True))
    axB.save(tpath_join("REFL"))


def test_reduce(test_trigger, tpath_join, plot, dprint):
    model = IFO_model.build_model(theta=+0.01)
    sys1 = model.sys1
    sys1c = sys1.copy()

    sys1.reduce()
    sys1.reduce()
    dprint(sys1.E[-2:, :])
    dprint(sys1.E[:, -2:])
    dprint(sys1.A[-2:, :])
    dprint(sys1.A[:, -2:])
    dprint(sys1.B[-2:, :])
    dprint(sys1.C[:-2, :])

    F_Hz = logspaced(1, model.FSR_Hz * 0.8, 1000)
    xfer_DARM = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )
    xfer_DARMc = sys1.xfer(
        F_Hz=F_Hz,
        iname="DARM.i0",
        oname="Msrm+A-oP",
    )

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_DARM))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
        axB.ax0.loglog(F_Hz, abs(xfer_DARMc))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARMc, deg=True))
        axB.save(tpath_join("DARM"))

    with test_trigger(trigger, plot=plot):
        np.testing.assert_almost_equal(xfer_DARM / xfer_DARMc, 1, decimal=5)
    return


def test_reducer2(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(theta=-0.01)
    sys1 = model.sys1
    sys1c = sys1.copy()
    # assert(sys1.A is sys1c.A)

    print(sys1.states.idx2name)
    sys1.reduce(
        method="diag",
        states=IFO_model.states_optical_i[:16],
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

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_DARM))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
        axB.ax0.loglog(F_Hz, abs(xfer_DARMc))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARMc, deg=True))
        axB.save(tpath_join("DARM"))

    with test_trigger(trigger, plot=plot):
        np.testing.assert_almost_equal(xfer_DARM / xfer_DARMc, 1, decimal=5)
    return


def test_percolate_names(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(theta=-0.01)
    sys1 = model.sys1
    sys1c = sys1.copy()

    # percolate back every other output name to see that it works
    sys1.percolate_names("output", sys1.output.idx2name[::2])

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

    def trigger(fail, plot):
        axB = mplfigB(Nrows=2)
        axB.ax0.loglog(F_Hz, abs(xfer_DARM))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARM, deg=True))
        axB.ax0.loglog(F_Hz, abs(xfer_DARMc))
        axB.ax1.semilogx(F_Hz, np.angle(xfer_DARMc, deg=True))
        axB.save(tpath_join("DARM"))

    with test_trigger(trigger, plot=plot):
        np.testing.assert_almost_equal(xfer_DARM / xfer_DARMc, 1, decimal=5)
    return


@pytest.mark.xfail(reason="Need to revisit these")
def test_controllable(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=4,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    # sys1.reduce(method = 'diag')
    # sys1.reduce(method = 'SVD')
    # sys1.permute_E_diagonal(location = 'upper left')
    # IFO_model.print_ssd(sys1)
    # sys1.controllable_staircase(reciprocal_system = True, do_reduce = True)
    # sys1.A = sys1.A[::-1, ::-1]
    # sys1.E = sys1.E[::-1, ::-1]
    # sys1.C = sys1.C[:, ::-1]
    # sys1.B = sys1.B[::-1, :]
    sys1.controllable_staircase(reciprocal_system=False)

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


@pytest.mark.xfail(reason="Need to revisit these")
def test_controllable_noQRPN(test_trigger, tpath_join, plot):
    model = IFO_model_noM.build_model(
        theta=-0.01,
        space_order=4,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    sys1c = sys1.copy()
    sys1.reduce(method="diag")
    # sys1.reduce(method = 'SVD')
    # sys1.permute_E_diagonal(location = 'upper left')
    # IFO_model.print_ssd(sys1)
    sys1.controllable_staircase(reciprocal_system=True, do_reduce=True)
    # sys1.A = sys1.A[::-1, ::-1]
    # sys1.E = sys1.E[::-1, ::-1]
    # sys1.C = sys1.C[:, ::-1]
    # sys1.B = sys1.B[::-1, :]
    sys1.controllable_staircase(reciprocal_system=False)

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


def test_save(test_trigger, tpath_join, plot):
    model = IFO_model.build_model(
        theta=-0.01,
        space_order=4,
        # no_QRPN = True,
    )
    sys1 = model.sys1
    print("Inputs: ", sys1.inputs.idx2name)
    print("Output: ", sys1.output.idx2name)
    print("Output[n]: ", sys1.output.idx2name.index("Msrm+A-oP"))
    wield.control.save(
        tpath_join("system.mat"),
        dict(
            A=sys1.A,
            B=sys1.B,
            C=sys1.C,
            D=sys1.D,
            E=sys1.E,
            inputs=sys1.inputs.idx2name,
            output=sys1.output.idx2name,
        ),
    )
