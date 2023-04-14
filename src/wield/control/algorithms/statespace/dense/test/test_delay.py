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
from wield.control.algorithms.statespace.dense import delay_algorithms, xfer_algorithms

import scipy.signal

c_m_s = 299792458


def print_ssd(ssd):
    print("B", ssd.B)
    print("A", ssd.A)
    print("E", ssd.E)
    print("C", ssd.C)
    print("D", ssd.D)


def test_delay(tpath_join, test_trigger):
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=2)

    for idx_ord in range(1, 7):
        arm1 = delay_algorithms.bessel_delay_ABCDE(delta_t, order=idx_ord)
        print_ssd(arm1)

        F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

        xfer = xfer_algorithms.ss2xfer(*arm1, F_Hz=F_Hz)

        axB.ax0.semilogx(F_Hz, abs(xfer), label="order {}".format(idx_ord))
        axB.ax1.plot(F_Hz, np.angle(xfer, deg=True))

    xfer_delay = np.exp(-2j * np.pi * F_Hz * delta_t)
    axB.ax1.plot(F_Hz, np.angle(xfer_delay, deg=True), color="magenta", ls="--")
    axB.ax1.axvline(1 / delta_t / 4)
    axB.ax1.axvline(2 / delta_t / 4)
    axB.ax1.axvline(3 / delta_t / 4)
    axB.ax1.axvline(4 / delta_t / 4)
    axB.ax0.legend()
    axB.save(tpath_join("test"))


def test_big_delay(tpath_join, test_trigger):
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=2)

    idx_ord = 100
    arm1 = delay_algorithms.bessel_delay_ABCDE(delta_t, order=idx_ord)
    print_ssd(arm1)

    F_Hz = np.linspace(0.00 / delta_t, 50 / delta_t, 1000)
    xfer = xfer_algorithms.ss2xfer(*arm1, F_Hz=F_Hz)

    axB.ax0.semilogx(F_Hz, abs(xfer), label="order {}".format(idx_ord))
    axB.ax1.plot(F_Hz, np.angle(xfer, deg=True))

    xfer_delay = np.exp(-2j * np.pi * F_Hz * delta_t)
    axB.ax1.plot(F_Hz, np.angle(xfer_delay, deg=True), color="magenta", ls="--")
    axB.ax1.axvline(1 / delta_t / 4)
    axB.ax1.axvline(2 / delta_t / 4)
    axB.ax1.axvline(3 / delta_t / 4)
    axB.ax1.axvline(4 / delta_t / 4)
    axB.ax0.legend()
    axB.save(tpath_join("test"))
