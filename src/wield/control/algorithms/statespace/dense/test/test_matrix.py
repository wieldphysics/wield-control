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
from wield.control.algorithms.statespace.dense import matrix_algorithms

import numpy.testing


pytestmark = pytest.mark.xfail(reason="Need to revisit these")


def test_QRH():

    N = 10
    M = np.random.rand(N, N)
    M = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        float,
    )

    eye = np.eye(M.shape[0], M.shape[1])

    R, [Q], [QT] = matrix_algorithms.QR(
        mat=M,
        mshadow=None,
        qmul=[eye],
        qAmul=[eye],
        pivoting=False,
        # method   = 'Householder',
        method="Givens",
        Rexact=False,
    )
    R2, [Q], [QT] = matrix_algorithms.QR(
        mat=M,
        mshadow=None,
        qmul=[eye],
        qAmul=[eye],
        pivoting=False,
        # method   = 'Householder',
        method="Givens",
        Rexact=True,
    )

    import tabulate

    print("near", tabulate.tabulate(R))
    print("exact", tabulate.tabulate(R2))
    print(tabulate.tabulate(Q))
    print(tabulate.tabulate(QT))

    numpy.testing.assert_almost_equal(Q @ Q.T, eye)
    numpy.testing.assert_almost_equal(Q @ QT, eye)
    numpy.testing.assert_almost_equal(Q @ R2, M)


def test_QRHpivot():

    N = 10
    M = np.random.rand(N, N)
    M = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        float,
    )

    eye = np.eye(M.shape[0], M.shape[1])

    R, [Q], [QT], P = matrix_algorithms.QR(
        mat=M,
        mshadow=None,
        qmul=[eye],
        qAmul=[eye],
        pivoting=True,
        # method   = 'Householder',
        method="Givens",
        Rexact=True,
    )

    import tabulate

    print(P)
    print(tabulate.tabulate(R))
    # print(tabulate.tabulate(Q))
    # print(tabulate.tabulate(QT))

    numpy.testing.assert_almost_equal(Q @ Q.T, eye)
    numpy.testing.assert_almost_equal(Q @ QT, eye)
    numpy.testing.assert_almost_equal((Q @ R)[:, P], M)
