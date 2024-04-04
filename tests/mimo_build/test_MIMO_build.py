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

from wield.pytest import (  # noqa: F401
    tjoin,
)


from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB

from wield.control import MIMO
from HSTS import HSTS_build


c_m_s = 299792458


def test_HSTS_WS():
    ABCD, iod = HSTS_build.getHSTSModel()

    ws_hsts = MIMO.MIMOStateSpace(
        ABCD,
        inout=iod,
    )

    axB = mplfigB(Nrows=1, Ncols=1)

    f_Hz = logspaced(0.01, 100, 1000)

    rs = ws_hsts.fresponse(f=f_Hz)

    print(ws_hsts.ss._p)

    axB.ax0.loglog(*rs.siso("P.m3.disp.L", "P.gnd.disp.L").fplot_mag)
    axB.ax0.loglog(*ws_hsts.siso("P.m3.disp.L", "P.gnd.disp.L").fresponse(f=f_Hz).fplot_mag)
    axB.save(tjoin("test"))
    return

