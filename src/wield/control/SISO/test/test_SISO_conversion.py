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


def bode(xfer, axm, axp, *, deg=True, **kw):
    if axm is not None:
        line, = axm.plot(
            xfer.f,
            abs(xfer.tf),
            **kw
        )
        kw.setdefault('color', line.get_color())

    if axp is not None:
        line, = axp.plot(
            xfer.f,
            np.angle(xfer.tf, deg=deg),
            **kw
        )
    return


def test_conversion(tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    axB = mplfigB(Nrows=2, Ncols=2)
    F_Hz = logspaced(.000001, 100, 1000)

    F_p = [-0.001 - 10j, -0.001 + 10j, -10]
    F_z = [0, 0, 0]
    F_k = 1

    filt = SISO.zpk((F_z, F_p, F_k), fiducial_rtol=1, angular=False)

    print(filt)

    xfer1 = filt.fresponse(f=F_Hz).tf
    axB.ax0.loglog(F_Hz, abs(xfer1), label="Direct ZPK")
    axB.ax1.semilogx(F_Hz, np.angle(xfer1, deg=True))

    #xfer2 = (filt * filt).fresponse(f=F_Hz).tf
    #axB.ax0.loglog(F_Hz, abs(xfer2), label="ZPK self product")
    #axB.ax1.semilogx(F_Hz, np.angle(xfer2, deg=True))

    filt_ss = filt.asSS * 4
    filt_ssi = 1/filt_ss
    xfer3 = filt_ss.fresponse(f=F_Hz).tf
    xfer3b = filt_ssi.fresponse(f=F_Hz).tf
    axB.ax0.loglog(F_Hz, abs(xfer3) / 4, label="ZPK2SS")
    axB.ax1.semilogx(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.loglog(F_Hz, abs(1/xfer3b) / 4, label="ZPK2SS inv")
    axB.ax1.semilogx(F_Hz, np.angle(1/xfer3b, deg=True))

    axB.ax0_1.semilogx(F_Hz, abs(xfer3/xfer1) / 4, label="ZPK2SS")
    axB.ax1_1.semilogx(F_Hz, np.angle(xfer3/xfer1, deg=True))
    axB.ax0_1.semilogx(F_Hz, abs(1/xfer3b/xfer1) / 4, label="ZPK2SS inv")
    axB.ax1_1.semilogx(F_Hz, np.angle(1/xfer3b/xfer1, deg=True))

    filt_zpk = filt_ss.asZPK
    filt_zpki = filt_ssi.asZPK
    xfer4 = filt_zpk.fresponse(f=F_Hz).tf
    xfer4b = (1/filt_zpk).fresponse(f=F_Hz).tf
    xfer4c = (filt_zpki).fresponse(f=F_Hz).tf
    axB.ax0.loglog(F_Hz, abs(xfer4) / 4, label="SS2ZPK")
    axB.ax1.semilogx(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.loglog(F_Hz, abs(1/xfer4b) / 4, label="SS2ZPK inv")
    axB.ax1.semilogx(F_Hz, np.angle(1/xfer4b, deg=True))
    axB.ax0.loglog(F_Hz, abs(1/xfer4c) / 4, label="SS2ZPK ss inv")
    axB.ax1.semilogx(F_Hz, np.angle(1/xfer4c, deg=True))

    axB.ax0_1.semilogx(F_Hz, abs(xfer4/xfer1) / 4, label="SS2ZPK")
    axB.ax1_1.semilogx(F_Hz, np.angle(xfer3/xfer1, deg=True))
    axB.ax0_1.semilogx(F_Hz, abs(1/xfer4b/xfer1) / 4, label="SS2ZPK inv")
    axB.ax1_1.semilogx(F_Hz, np.angle(1/xfer4b/xfer1, deg=True))
    axB.ax0_1.semilogx(F_Hz, abs(1/xfer4c/xfer1) / 4, label="SS2ZPK ss inv")
    axB.ax1_1.semilogx(F_Hz, np.angle(1/xfer4c/xfer1, deg=True))

    axB.ax0.legend()
    axB.save(tpath_join("test_ZPK"))

    np.testing.assert_almost_equal(xfer3, 4 * xfer1)
    np.testing.assert_almost_equal(xfer4, 4 * xfer1)
    np.testing.assert_almost_equal(xfer3, 1/xfer3b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4c)


