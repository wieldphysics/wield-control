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

from .HSTS import HSTS_build

from wield.control import MIMO


def test_HSTS_WS(tpath_join):
    ABCD, iod = HSTS_build.getHSTSModel()

    ws_hsts = MIMO.statespace(
        ABCD,
        inout=iod,
    )

    axB = mplfigB(Nrows=1, Ncols=1)

    f_Hz = logspaced(0.01, 100, 1000)

    rs = ws_hsts.fresponse(f=f_Hz)

    print(ws_hsts.ss._p)

    axB.ax0.loglog(*rs.siso("P.m3.disp.L", "P.gnd.disp.L").fplot_mag)
    axB.ax0.loglog(*ws_hsts.siso("P.m3.disp.L", "P.gnd.disp.L").fresponse(f=f_Hz).fplot_mag)
    axB.save(tpath_join("test"))
    return

