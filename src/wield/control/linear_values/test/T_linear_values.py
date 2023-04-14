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
import wield.control.linear_values as lv

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


def T_linear_values(dprint, tpath_join, fpath_join):
    ZERO = lv.scalar(0)
    IDENT = lv.scalar(1)
    SCALAR = lv.scalar(2)
    SCALARa = lv.scalar(np.arange(2))

    DIAG = lv.diagonal([2, 4])
    DIAGa = lv.diagonal([2, 4 + np.arange(2)])

    MAT = lv.matrix(
        [[0, 1],
        [-1, 0]]
    )

    MATa = lv.matrix(
        [[0, 1],
        [-1, np.arange(2)]]
    )
    print("TEST", MATa.value)

    print(MAT @ MAT)
    assert(np.all(MAT @ MAT == lv.matrix(
        [[-1, 0],
         [0, -1]])
    ))
    print(DIAG @ MAT)
    assert(np.all(DIAG @ MAT == lv.matrix(
        [[0, 2],
         [-4, 0]])
    ))
    print(MAT @ DIAG)
    assert(np.all(MAT @ DIAG == lv.matrix(
        [[0, 4],
         [-2, 0]])
    ))
    print(DIAG @ DIAG)
    assert(np.all(DIAG @ DIAG == lv.diagonal(
        [4, 16]
    )))

    print(DIAG @ SCALAR)
    print(SCALAR @ DIAG)
    assert(np.all(DIAG @ SCALAR == lv.diagonal(
        [4, 8]
    )))

    print(MAT @ SCALAR)
    print(SCALAR @ MAT)
    assert(np.all(MAT @ SCALAR == lv.matrix(
        [[0, 2],
         [-2, 0]])
    ))

    print('--------------')
    # now the array style
    print("A", MATa @ MAT)
    print("B", MAT @ MATa)
    print("C", MATa @ MATa)
    assert(np.all(MAT @ MAT == lv.matrix(
        [[-1, 0],
         [0, -1]])
    ))
    print("A", DIAGa @ MAT)
    print("B", MAT @ DIAGa)
    print("C", MATa @ DIAG)
    print("D", DIAG @ MATa)
    print("E", DIAGa @ MATa)
    print("F", MATa @ DIAGa)

    assert(np.all(DIAG @ MAT == lv.matrix(
        [[0, 2],
         [-4, 0]])
    ))
    assert(np.all(MAT @ DIAG == lv.matrix(
        [[0, 4],
         [-2, 0]])
    ))

    print("G", DIAGa @ DIAG)
    print("H", DIAG @ DIAGa)
    assert(np.all(DIAG @ DIAG == lv.diagonal(
        [4, 16]
    )))

    print("I", DIAGa @ SCALAR)
    print("J", SCALAR @ DIAGa)
    assert(np.all(DIAG @ SCALAR == lv.diagonal(
        [4, 8]
    )))

    print("K", MATa @ SCALAR)
    print("L", SCALAR @ MATa)
    assert(np.all(MAT @ SCALAR == lv.matrix(
        [[0, 2],
         [-2, 0]])
    ))

    print('++++++++++++++')
    # now the array style
    print("A", MAT + MAT)
    assert(np.all(MAT + MAT == lv.matrix(
        [[0, 2],
         [-2, 0]])
    ))

    print("B", DIAG + MAT)
    assert(np.all(DIAG + MAT == lv.matrix(
        [[2, 1],
         [-1, 4]])
    ))
    print(MAT + DIAG)
    assert(np.all(MAT + DIAG == lv.matrix(
        [[2, 1],
         [-1, 4]])
    ))
    print(DIAG + DIAG)
    assert(np.all(DIAG + DIAG == lv.diagonal(
        [4, 8]
    )))

    print(DIAG + SCALAR)
    print(SCALAR + DIAG)
    assert(np.all(DIAG + SCALAR == lv.diagonal(
        [4, 6]
    )))

    print(MAT + SCALAR)
    print(SCALAR + MAT)
    assert(np.all(MAT + SCALAR == lv.matrix(
        [[2, 1],
         [-1, 2]])
    ))

    print('--------------')
    # now the array style
    print("A", MATa + MAT)
    print("B", MAT + MATa)
    print("C", MATa + MATa)
    print("A", DIAGa + MAT)
    print("B", MAT + DIAGa)
    print("C", MATa + DIAG)
    print("D", DIAG + MATa)
    print("E", DIAGa + MATa)
    print("F", MATa + DIAGa)

    assert(np.all(DIAG + MATa == lv.matrix([
        [(2, 2), (1, 1)],
        [(-1, -1), (4, 5)]
    ])))
    assert(np.all(DIAGa + MAT == lv.matrix([
        [(2, 2), (1, 1)],
        [(-1, -1), (4, 5)]
    ])))

    print("G", DIAGa + DIAG)
    print("H", DIAG + DIAGa)
    assert(np.all(DIAGa + DIAG == lv.diagonal(
        [(4, 4), (8, 9)]
    )))
    assert(np.all(DIAG + DIAGa == lv.diagonal(
        [(4, 4), (8, 9)]
    )))

    print("I", DIAGa + SCALAR)
    print("J", SCALAR + DIAGa)
    assert(np.all(DIAGa + SCALAR == lv.diagonal(
        [(4, 4), (6, 7)]
    )))

    print("K", MATa + SCALAR)
    print("L", SCALAR + MATa)
    assert(np.all(MATa + SCALAR == lv.matrix(
        [[(2, 2), (1, 1)],
         [(-1, -1), (2, 3)]])
    ))
    print("M", MAT + SCALARa)
    print("N", SCALARa + MAT)
    assert(np.all(MAT + SCALARa == lv.matrix(
        [[(0, 1), (1, 1)],
         [(-1, -1), (0, 1)]])
    ))
