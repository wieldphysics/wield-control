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


def ss2xfer(A, B, C, D, E=None, F_Hz=None, idx_in=None, idx_out=None):
    return ss2response_siso(
        A, B, C, D, E, s = 2j * np.pi * F_Hz, idx_in=idx_in, idx_out=idx_out
    )


def ss2response_siso(A, B, C, D, E=None, sorz=None, idx_in=None, idx_out=None):
    sorz = np.asarray(sorz)

    if idx_in is None:
        if B.shape[1] == 1:
            idx_in = 0
        else:
            raise RuntimeError("Must specify idx_in if B indicates MISO/MIMO system")
    if idx_out is None:
        if C.shape[0] == 1:
            idx_out = 0
        else:
            raise RuntimeError("Must specify idx_in if C indicates SIMO/MIMO system")

    B = B[:, idx_in: idx_in + 1]
    C = C[idx_out: idx_out + 1, :]
    D = D[idx_out: idx_out + 1, idx_in: idx_in + 1]
    return ss2response_mimo(A, B, C, D, E, sorz=sorz)[..., 0, 0]


def ss2response_mimo(A, B, C, D, E=None, sorz=None):
    sorz = np.asarray(sorz)

    if A.shape[-2:] == (0, 0):
        print("BCAST", A.shape, D.shape)
        return np.broadcast_to(D, sorz.shape + D.shape[-2:])

    if E is None:
        S = np.eye(A.shape[0]).reshape(1, *A.shape) * sorz.reshape(-1, 1, 1)
        return (
            np.matmul(C, np.matmul(np.linalg.inv(S - A.reshape(1, *A.shape)), B)) + D
        )
    else:
        return (
            np.matmul(
                C,
                np.matmul(
                    np.linalg.inv(E * sorz.reshape(-1, 1, 1) - A.reshape(1, *A.shape)),
                    B
                ),
            ) + D
        )
