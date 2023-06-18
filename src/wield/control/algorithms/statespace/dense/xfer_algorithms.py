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
import scipy.linalg


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
        # print("BCAST", A.shape, D.shape)
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


def ss2response_laub(A, B, C, D, E=None, sorz=None):
    """
    Use Laub's method to calculate the frequency response. Very fast but in some cases less numerically stable.
    Generally OK if the A/E matrix has been balanced first.
    """
    sorz = np.asarray(sorz)

    if A.shape[-2:] == (0, 0):
        # print("BCAST", A.shape, D.shape)
        return np.broadcast_to(D, sorz.shape + D.shape[-2:])

    if E is not None:
        if np.all(E == np.eye(E.shape[-1])):
            E = None

    if E is None:
        A, Z = scipy.linalg.schur(A, output='complex')
        B = Z.transpose().conjugate() @ B
        C = C @ Z

        diag = (np.diag(A).reshape(1, -1) - sorz.reshape(-1, 1))

        retval = array_solve_triangular(-A, -diag, B)

        # retval2 = np.linalg.inv(np.eye(A.shape[-1]) * sorz.reshape(-1, 1, 1) - A.reshape(1, *A.shape)) @ B
        # print(retval - retval2)
        return C @ retval + D

        return (
                C @ (
                    np.linalg.inv(np.eye(A.shape[-1]) * sorz.reshape(-1, 1, 1) - A.reshape(1, *A.shape))
                    @ B
                )
            ) + D
    else:
        # TODO, E matrix not supported yet
        import warnings
        warnings.warn("Laub method used on descriptor system. Not supported yet (using slow Horner method fallback)")
        return (
                C @ (
                    np.linalg.inv(E * sorz.reshape(-1, 1, 1) - A.reshape(1, *A.shape))
                    @ B
                )
            ) + D


def array_solve_triangular(A, D, b):
    """
    Solve a triangular matrix system.
    A is a (M, M). D is (..., M) are broadcasted diagonals, and b is (M, N)
    """
    # b = np.eye(b.shape[-2], dtype=complex)
    # b = b[:, -2:-1]
    # idx = -2
    # D = D[idx:idx+1]

    bwork = np.broadcast_to(b, D.shape[:-1] + b.shape).copy()

    M = A.shape[-1]

    # print("A", A.shape)
    # print("D", D.shape)
    # print("B", b.shape)
    # print("Bwork", bwork.shape)

    # needed for broadcast magic to preserve the size of .shape after indexing
    # Dv = D.reshape(*D.shape, 1)
    # Av = A.reshape(*A.shape, 1)

    for idx in range(M - 1, -1, -1):
        bwork[..., idx, :] /= D[..., idx:idx+1]
        bwork[..., :idx, :] -= A[:idx, idx:idx+1] * bwork[..., idx:idx+1, :]

    #Atest = A - np.diag(np.diag(A)) + np.diag(D)

    # Atest = A - np.diag(np.diag(A)) + np.diag(D[idx])
    # bwork2 = np.linalg.inv(Atest) @ b
    # print(abs(Atest @ bwork - b) < 1e-6)
    # print(abs(Atest @ bwork[idx] - Atest @ bwork2) < 1e-6)
    return bwork








