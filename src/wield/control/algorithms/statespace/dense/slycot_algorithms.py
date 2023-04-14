#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

from wield import declarative
import numpy as np

import slycot


def rescale_slycot(A, B, C, D, E):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    out = slycot.transform.tb01id(
        n,
        m,
        p,
        10,
        np.copy(A),
        np.copy(B),
        np.copy(C),
        job="A",
    )
    s_norm, A, B, C, scale = out
    return wield.bunch.Bunch(
        ABCDE=(A, B, C, D, E),
        s_norm=s_norm,
        scale=scale,
    )
