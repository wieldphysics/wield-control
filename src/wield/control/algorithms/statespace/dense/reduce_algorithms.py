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
import collections

from . import shuffle_algorithms


def nonzero_test(val):
    return val != 0


def rank_nonzero(val):
    if val == 1:
        return (2,)
    elif val == -1:
        return (1,)
    elif val > 1:
        return (0, 2)
    elif val < 1:
        return (0, 1)
    else:
        return (0, 0, 0)


def reduce_diag_inplace(
    ABCDE,
    Nst,
    Nco,
    tol=1e-10,
    nonzero_test=nonzero_test,
    rank_nonzero=rank_nonzero,
):
    A, B, C, D, E = ABCDE
    Ist = A.shape[1] - Nst
    Ico = A.shape[0] - Nco

    assert not np.any(E[Ico:, :])
    assert not np.any(E[:, Ist:])

    nonzero_test = np.frompyfunc(nonzero_test, 1, 1)
    subA = A[Ico:, Ist:]
    subAz = nonzero_test(subA)
    countCo = np.sum(subAz, axis=-1)
    countSt = np.sum(subAz, axis=-2)
    # print(countSt, countCo)
    diagd = collections.defaultdict(list)
    for idx_co in np.argwhere(countCo == 1)[:, 0]:
        # print(idx_co)
        idx_st = np.argwhere(subAz[idx_co, :])[0][0]
        # print(idx_st)
        val = subA[idx_co, idx_st]
        # print(idx_co, idx_st, val)
        Ncross = countSt[idx_st]
        # print(idx_co, idx_st, Ncross, val, rank_nonzero(val))
        diagd[idx_st].append((rank_nonzero(val), Ncross, idx_co, val))
    st_ranges = []
    co_ranges = []
    diag = []
    for idx_st, co_list in sorted(diagd.items()):
        co_list.sort()
        rank, Ncross, idx_co, val = co_list[-1]
        # print("IDXs", idx_co, idx_st)
        st_ranges.append((Ist + idx_st, Ist + idx_st + 1))
        co_ranges.append((Ico + idx_co, Ico + idx_co + 1))
        diag.append(val)
    diag = np.asarray(diag)
    Nlist = len(st_ranges)
    shuffle_algorithms.percolate_inplace(
        A=A, B=B, C=C, D=D, E=E, which="states", ranges=st_ranges
    )
    shuffle_algorithms.percolate_inplace(
        A=A, B=B, C=C, D=D, E=E, which="constr", ranges=co_ranges
    )
    # print(A[-Nlist:, -Nlist:])
    # print(diag)

    # remap to consider only the remaining pieces
    Ist = A.shape[1] - Nlist
    Ico = A.shape[0] - Nlist
    A12mod = A[:Ico, Ist:] / diag.reshape(1, -1)
    A[:Ico, :Ist] -= A12mod @ A[Ico:, :Ist]
    B[:Ico, :] -= A12mod @ B[Ico:, :]

    C2mod = C[:, Ist:] / diag.reshape(1, -1)
    C[:, :Ist] -= C2mod @ A[Ico:, :Ist]
    D -= C2mod @ B[Ico:, :]

    C = C[:, :Ist]
    B = B[:Ico, :]
    A = A[:Ico, :Ist]
    E = E[:Ico, :Ist]
    return (A, B, C, D, E), Nlist


def reduce_SVD_inplace(ABCDE, Nst, Nco, tol=1e-10):
    A, B, C, D, E = ABCDE
    Ist = A.shape[1] - Nst
    Ico = A.shape[0] - Nco

    assert Nst == Nco
    assert not np.any(E[Ico:, :])
    assert not np.any(E[:, Ist:])

    A22 = A[Ico:, Ist:]
    U, S, V = scipy.linalg.svd(A22)

    # now reorder for smallest first
    S = S[::-1]
    Nlast = len(S) - np.searchsorted(S, S[-1] * tol)
    # print("NLAST", Nlast)
    # redefines constr
    U = U[:, ::-1]
    Ui = U.T
    # redefines states
    V = V[::-1, :]
    Vi = V.T

    # print(S)
    # print(np.diagonal(mat))
    B[Ico:, :] = Ui @ B[Ico:, :]
    C[:, Ist:] = C[:, Ist:] @ Vi
    # A22
    # one might expect to use diag(S) here, but
    # the svd is still unstable, so we need to use only orthogonal
    # transformations
    if Nlast == len(S):
        A[Ico:, Ist:] = np.diag(S)
    else:
        A[Ico:, Ist:] = Ui @ A[Ico:, Ist:] @ Vi
    # A12
    A[:Ico, Ist:] = A[:Ico, Ist:] @ Vi
    # A21
    A[Ico:, :Ist] = Ui @ A[Ico:, :Ist]

    # now, select the last few rows that are OK
    # and reduce them down
    S = S[-Nlast:]
    Ist = A.shape[1] - Nlast
    Ico = A.shape[0] - Nlast

    A22 = A[Ico:, Ist:]
    if False:
        # this secondary SVD is useful to improve
        # numerical stability yet further, but appears to be unnecessary
        U, S, V = scipy.linalg.svd(A22)
        # now reorder for smallest first
        # redefines constr
        Ui = U.T
        # redefines states
        Vi = V.T
        # print(S)
        B[Ico:, :] = Ui @ B[Ico:, :]
        C[:, Ist:] = C[:, Ist:] @ Vi

        B[Ico:, :] = Ui @ B[Ico:, :]
        C[:, Ist:] = C[:, Ist:] @ Vi
        # A12
        A[:Ico, Ist:] = A[:Ico, Ist:] @ Vi
        # A21
        A[Ico:, :Ist] = Ui @ A[Ico:, :Ist]

    A12mod = A[:Ico, Ist:] / S.reshape(1, -1)
    A[:Ico, :Ist] -= A12mod @ A[Ico:, :Ist]
    B[:Ico, :] -= A12mod @ B[Ico:, :]

    C2mod = C[:, Ist:] / S.reshape(1, -1)
    C[:, :Ist] -= C2mod @ A[Ico:, :Ist]
    D -= C2mod @ B[Ico:, :]

    C = C[:, :Ist]
    B = B[:Ico, :]
    A = A[:Ico, :Ist]
    E = E[:Ico, :Ist]
    return (A, B, C, D, E), Nlast


def rank_nonzeroE(val, N1, N2):
    return N1 * N2


def permute_Ediag_inplace(
    ABCDE,
    Nst=None,
    Nco=None,
    location="upper left",
    nonzero_test=nonzero_test,
    rank_nonzeroE=rank_nonzeroE,
):
    """
    Permutes the E matrix to be on the location='upper left' or 'upper right'
    and as diagonal as possible.

    return ABCDE, and then Ndiag, Ndiag + NcoM, Ndiag + NstM
    which are the boundaries of the diagonal block and the nondiagonal nonzero block

    TODO, should output the boundary
    TODO, algorithm could use swaps rather than percolates
    """
    A, B, C, D, E = ABCDE
    if Nst is None:
        Nst = A.shape[1]
    if Nco is None:
        Nco = A.shape[0]
    Ist = A.shape[1] - Nst
    Ico = A.shape[0] - Nco

    nonzero_test = np.frompyfunc(nonzero_test, 1, 1)
    subE = E[Ico:, Ist:]
    subEz = nonzero_test(subE)
    countCo = np.sum(subEz, axis=-1)
    countSt = np.sum(subEz, axis=-2)
    # print(countSt, countCo)
    diagd = collections.defaultdict(list)
    co0 = []
    coM = []
    st0 = []
    stM = []
    for idx_co in range(len(countCo)):
        N1 = countCo[idx_co]
        if N1 == 0:
            co0.append(idx_co)
        elif N1 == 1:
            # print(idx_co)
            idx_st = np.argwhere(subEz[idx_co, :])[0][0]
            # print(idx_st)
            val = subE[idx_co, idx_st]
            # print(idx_co, idx_st, val)
            N2 = countSt[idx_st]
            # print(idx_co, idx_st, Ncross, val, rank_nonzero(val))
            diagd[idx_st].append((rank_nonzeroE(val, N1, N2), N1, N2, idx_co, val))
        else:
            coM.append(idx_co)
    for idx_st in range(len(countSt)):
        N1 = countSt[idx_st]
        if N1 == 0:
            st0.append(idx_st)
        elif N1 == 1:
            # already in diagd
            pass
        else:
            stM.append(idx_st)
    st_ranges = []
    co_ranges = []
    for idx_st, co_list in sorted(diagd.items()):
        co_list.sort()
        rank, N1, N2, idx_co, val = co_list[-1]
        # print("IDXs", idx_co, idx_st)
        st_ranges.append((Ist + idx_st, Ist + idx_st + 1))
        co_ranges.append((Ico + idx_co, Ico + idx_co + 1))
    Ndiag = len(diagd)
    # also add the many-E lines
    for idx_co in coM:
        co_ranges.append((Ico + idx_co, Ico + idx_co + 1))
    NcoM = len(coM)
    for idx_st in stM:
        st_ranges.append((Ist + idx_st, Ist + idx_st + 1))
    NstM = len(coM)
    # and the zero lines at the end
    for idx_co in co0:
        co_ranges.append((Ico + idx_co, Ico + idx_co + 1))
    for idx_st in st0:
        st_ranges.append((Ist + idx_st, Ist + idx_st + 1))

    location = location.lower()
    if location == "upper left":
        # reverse since percolation puts things at the end
        co_ranges = co_ranges[::-1]
        st_ranges = st_ranges[::-1]
    elif location == "upper right":
        co_ranges = co_ranges[::-1]
    else:
        raise RuntimeError("Unrecognized Location")
    shuffle_algorithms.percolate_inplace(
        A=A, B=B, C=C, D=D, E=E, which="states", ranges=st_ranges
    )
    shuffle_algorithms.percolate_inplace(
        A=A, B=B, C=C, D=D, E=E, which="constr", ranges=co_ranges
    )
    if location == "upper left":
        # reverse again to move things at end back to the beginning
        A = A[::-1, ::-1]
        E = E[::-1, ::-1]
        B = B[::-1, :]
        C = C[:, ::-1]
    elif location == "upper right":
        A = A[::-1, :]
        E = E[::-1, :]
        B = B[::-1, :]

    return (A, B, C, D, E), Ndiag, Ndiag + NcoM, Ndiag + NstM
