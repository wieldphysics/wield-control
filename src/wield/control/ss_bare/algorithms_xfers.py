#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Algorithms for calculating xfer functions
"""

from ..utilities import algorithm_choice
from ..algorithms.statespace.dense import xfer_algorithms


def ss2fresponse_laub(ss, sorz, **kwargs):
    # TODO fix this import

    # the balanceABC does a much better job than just balancing A
    # ss = ss.balanceA()
    ss = ss.balanceABC(which='ABC')
    return xfer_algorithms.ss2response_laub(
        A=ss.A,
        B=ss.B,
        C=ss.C,
        D=ss.D,
        E=ss.E,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_laub', ss2fresponse_laub, 100)


def ss2fresponse_horner(ss, sorz, **kwargs):
    # TODO fix this import
    return xfer_algorithms.ss2response_mimo(
        A=ss.A,
        B=ss.B,
        C=ss.C,
        D=ss.D,
        E=ss.E,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_horner', ss2fresponse_horner, 50)


def ss2fresponse_testing(ss, sorz, **kwargs):
    # TODO fix this import
    # ss = ss.balanceA()
    ss = ss.balanceABC(which='ABC')
    return xfer_algorithms.ss2response_laub_testing(
        A=ss.A,
        B=ss.B,
        C=ss.C,
        D=ss.D,
        E=ss.E,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_testing', ss2fresponse_testing, 80)

