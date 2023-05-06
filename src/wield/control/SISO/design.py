#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions for designing and synthesizing SISO filters
"""
import scipy.signal
import numpy as np
from .zpk import zpk


def delay_thiran_raw(delay_s, order=1):
    """
    Create a thiran filter to simulate a delay line with maximally flat group delay for a given filter order.

    This is done by generating the poles of a Bessel filter - which has maximally flat group delay for a low-pass filter.
    The poles are then rescaled to generate the desired delay and then right-plane zeroes are created to mirror the poles.
    This generates an all-pass filter with constant gain and constant delay. This implementation can create filters to very
    high order of 100 poles or more.
    """
    # take the poles of this normalized bessel filter (delay=1s)
    z, p, k = scipy.signal.besselap(order, norm="delay")

    # now rescale for desired delay
    roots = p / delay_s * 2
    if order % 2 == 0:
        k = 1
    else:
        k = -1

    return zpk(
        -roots.conjugate(),
        roots,
        k
    )


def root_factored_quadrature_sum(*filts):
    """
    Return a square root filter

    TODO, make this numerically better behaved and test its output.

    This should actually be implemented using spectral factorization using and ARE.
    But debugging that sounds hard. This should be similarly robust but is not a general method.
    """
    ss_sq = None
    for filt in filts:
        filt = filt.asSS
        if ss_sq is None:
            ss_sq = filt.conjugate() * filt
        else:
            ss_sq = ss_sq + filt.conjugate() * filt

    # convert back to ZPK
    ss_sqZPK = ss_sq.asZPK

    def stable_root_extract(roots):
        # This is not a fully robust way to do this!
        roots = np.asarray(roots)
        lhp = roots[roots.real < 0]
        eq0 = roots[roots.real == 0]
        rhp = roots[roots.real > 0]
        assert (len(lhp) == len(rhp))
        assert (len(eq0) % 2 == 0)
        eq0 = sorted(eq0, key=lambda r: abs(r.imag))

        # skip every other real one
        return list(lhp) + list(eq0[::4]) + list(eq0[1::4])
    p = stable_root_extract(ss_sqZPK.p)
    z = stable_root_extract(ss_sqZPK.z)
    k = ss_sqZPK.k**0.5

    ss_rt = zpk(z, p, k, fiducial_f=[])

    # TODO, test against the actual magnitude

    return ss_rt
