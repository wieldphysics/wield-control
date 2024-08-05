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


def ss2fresponse_laub(rss, sorz, **kwargs):
    # TODO fix this import

    # the balanceABC does a much better job than just balancing A
    # rss = rss.balanceA()
    rss = rss.balanceABC(which='ABC')
    return xfer_algorithms.ss2response_laub(
        A=rss.A,
        B=rss.B,
        C=rss.C,
        D=rss.D,
        E=rss.e,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_laub', ss2fresponse_laub, 100)


def ss2fresponse_horner(rss, sorz, **kwargs):
    # TODO fix this import
    return xfer_algorithms.ss2response_mimo(
        A=rss.A,
        B=rss.B,
        C=rss.C,
        D=rss.D,
        E=rss.e,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_horner', ss2fresponse_horner, 50)


def ss2fresponse_testing(rss, sorz, **kwargs):
    # TODO fix this import
    # rss = rss.balanceA()
    rss = rss.balanceABC(which='ABC')
    return xfer_algorithms.ss2response_laub_testing(
        A=rss.A,
        B=rss.B,
        C=rss.C,
        D=rss.D,
        E=rss.e,
        sorz=sorz,
        **kwargs
    )


algorithm_choice.algorithm_register('ss2fresponse', 'ss2fresponse_testing', ss2fresponse_testing, 80)

