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

c_m_s = 299792458


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


def bessel_delay_ZPK(delay_s, order=1, rescale=None):
    # take the poles of this normalized bessel filter (delay=1s)
    z, p, k = scipy.signal.besselap(order, norm="delay")

    # now rescale for desired delay
    roots = p / delay_s * 2
    if order % 2 == 0:
        k = 1
    else:
        k = -1

    return SISO.zpk(
        -roots.conjugate(),
        roots,
        k
    )


def test_ZPK_delay_many(tpath_join):
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=2)
    F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

    for idx_ord in range(1, 7):
        filt = bessel_delay_ZPK(delta_t, order=idx_ord)

        xfer = filt.fresponse(f=F_Hz).tf
        # print("filt.z: ", filt.z, tuple(filt.z))
        # print("filt.p: ", filt.p, tuple(filt.p))
        w, zpk0 = scipy.signal.freqs_zpk(
            tuple(filt.z),
            tuple(filt.p),
            filt.k,
            2 * np.pi * F_Hz
        )

        # Test that the ZPK is internally using the scipy convention
        np.testing.assert_almost_equal(xfer, zpk0)

        axB.ax0.semilogx(F_Hz, abs(xfer), label="order {}".format(idx_ord))
        axB.ax1.plot(F_Hz, np.angle(xfer, deg=True))

    xfer_delay = np.exp(-2j * np.pi * F_Hz * delta_t)
    axB.ax1.plot(F_Hz, np.angle(xfer_delay, deg=True), color="magenta", ls="--")
    axB.ax1.axvline(1 / delta_t / 4, ls='--', color='black')
    axB.ax1.axvline(2 / delta_t / 4, ls='--', color='black')
    axB.ax1.axvline(3 / delta_t / 4, ls='--', color='black')
    axB.ax1.axvline(4 / delta_t / 4, ls='--', color='black')
    axB.ax0.legend()
    axB.save(tpath_join("test_ZPK"))


@pytest.mark.parametrize('idx_ord', [4, 40, 100])
def test_ZPK_delay_various(idx_ord, tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=2)
    F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

    filt = bessel_delay_ZPK(delta_t, order=idx_ord)

    xfer1 = filt.fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer1), label="Direct ZPK")
    axB.ax1.plot(F_Hz, np.angle(xfer1, deg=True))

    xfer2 = (filt * filt).fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer2), label="ZPK self product")
    axB.ax1.plot(F_Hz, np.angle(xfer2, deg=True))

    filt_ss = filt.asSS * 4
    filt_ssi = 1/filt_ss
    xfer3 = filt_ss.fresponse(f=F_Hz).tf
    xfer3b = filt_ssi.fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer3) / 4, label="ZPK2SS")
    axB.ax1.plot(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer3b) / 4, label="ZPK2SS inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer3b, deg=True))

    filt_zpk = filt_ss.asZPK
    filt_zpki = filt_ssi.asZPK
    xfer4 = filt_zpk.fresponse(f=F_Hz).tf
    xfer4b = (1/filt_zpk).fresponse(f=F_Hz).tf
    xfer4c = (filt_zpki).fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer4) / 4, label="SS2ZPK")
    axB.ax1.plot(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer4b) / 4, label="SS2ZPK inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer4b, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer4c) / 4, label="SS2ZPK ss inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer4c, deg=True))

    axB.ax0.legend()
    axB.save(tpath_join("test_ZPK"))

    np.testing.assert_almost_equal(xfer2, xfer1**2)
    np.testing.assert_almost_equal(xfer3, 4 * xfer1)
    np.testing.assert_almost_equal(xfer4, 4 * xfer1)
    np.testing.assert_almost_equal(xfer3, 1/xfer3b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4c)


@pytest.mark.parametrize('zpk', [
    ((0,), (), 0.1),
    ((), (0,), 0.1),
    ((-1,), (), 0.1),
    ((), (-1,), 0.1),
    ((0, 0,), (10, 10), 0.1),
    ((10, 10), (0, 0), 0.1),
    ((-1, -1,), (10, 10), 0.1),
    ((10, 10), (-1, -1), 0.1),
    ((0, ), (10, 10), 0.1),
    ((10, 10), (0,), 0.1),
    ((-1, ), (10, 10), 0.1),
    ((10, 10), (-1, ), 0.1),
])
def test_ZPK_various(zpk, tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    delta_t = 1
    axB = mplfigB(Nrows=2)
    F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

    filt = SISO.zpk(zpk, fiducial_rtol=1e-7)
    print(filt)

    xfer1 = filt.fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer1), label="Direct ZPK")
    axB.ax1.plot(F_Hz, np.angle(xfer1, deg=True))

    xfer2 = (filt * filt).fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer2), label="ZPK self product")
    axB.ax1.plot(F_Hz, np.angle(xfer2, deg=True))

    filt_ss = filt.asSS * 4
    filt_ssi = 1/filt_ss
    xfer3 = filt_ss.fresponse(f=F_Hz).tf
    xfer3b = filt_ssi.fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer3) / 4, label="ZPK2SS")
    axB.ax1.plot(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer3b) / 4, label="ZPK2SS inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer3b, deg=True))

    filt_zpk = filt_ss.asZPK
    filt_zpki = filt_ssi.asZPK
    xfer4 = filt_zpk.fresponse(f=F_Hz).tf
    xfer4b = (1/filt_zpk).fresponse(f=F_Hz).tf
    xfer4c = (filt_zpki).fresponse(f=F_Hz).tf
    axB.ax0.semilogx(F_Hz, abs(xfer4) / 4, label="SS2ZPK")
    axB.ax1.plot(F_Hz, np.angle(xfer3, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer4b) / 4, label="SS2ZPK inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer4b, deg=True))
    axB.ax0.semilogx(F_Hz, abs(1/xfer4c) / 4, label="SS2ZPK ss inv")
    axB.ax1.plot(F_Hz, np.angle(1/xfer4c, deg=True))

    axB.ax0.legend()
    axB.save(tpath_join("test_ZPK"))

    np.testing.assert_almost_equal(xfer2, xfer1**2)
    np.testing.assert_almost_equal(xfer3, 4 * xfer1)
    np.testing.assert_almost_equal(xfer4, 4 * xfer1)
    np.testing.assert_almost_equal(xfer3, 1/xfer3b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4b)
    np.testing.assert_almost_equal(xfer4, 1/xfer4c)

def test_ZPK_delay_math(tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=1)
    F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

    filt = bessel_delay_ZPK(delta_t, order=50)

    fr = filt.fresponse(f=F_Hz)
    axB.ax0.semilogx(*fr.fplot_mag, label="SS2ZPK")
    #axB.ax1.semilogx(fr.fplot_deg135, label="SS2ZPK")

    fr2 = (1 / (filt.asSS)).fresponse(f=F_Hz)
    #
    fr2 = (1 / (1 - .9 * filt.asSS)).fresponse(f=F_Hz)
    axB.ax0.semilogx(*fr2.fplot_mag, label="SS2ZPK")
    #axB.ax1.semilogx(fr2.fplot_deg, label="SS2ZPK")

    axB.ax0.legend()
    axB.save(tpath_join("test_ZPK.pdf"))


@pytest.mark.parametrize('idx_ord', [4, 40, 100])
def test_SS_delay_print(idx_ord, tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    length_m = 3995
    delta_t = length_m / c_m_s
    delta_t = 1
    axB = mplfigB(Nrows=2)
    F_Hz = logspaced(0.01 / delta_t, 2 / delta_t, 1000)

    filt = bessel_delay_ZPK(delta_t, order=idx_ord).asSS

    print()
    print()
    filt.print_nonzero()
