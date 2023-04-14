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


