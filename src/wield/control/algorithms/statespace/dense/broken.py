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
import scipy.linalg

from .eig_algorithms import eigspaces_right_real


def controllable_staircase(
    A,
    B,
    C,
    D,
    E,
    tol=1e-7,
):
    """
    An attempt to do the staircase form using more general QR transforms for
    speed, not faster and very numerically aweful so far
    """
    from icecream import ic
    import tabulate

    Ninputs = B.shape[1]
    Nstates = A.shape[0]
    Nconstr = A.shape[1]
    Noutput = C.shape[0]

    print("E", tabulate.tabulate(abs(E[:9, :9])))
    BA, E = scipy.linalg.qr_multiply(E, np.hstack([B, A]), pivoting=False, mode="left")
    print(E)
    B = BA[:, :Ninputs]
    A = BA[:, Ninputs:]
    del BA

    if False:
        # test stability
        BET, A, P = scipy.linalg.qr_multiply(
            A,
            np.hstack([B, E]).T,
            pivoting=True,
            conjugate=True,
            mode="right",
        )
        BE = BET.T
        B = BE[:, :Ninputs]
        E = BE[:, Ninputs:]
        E = E[:, P]
        C = C[:, P]
        del BE
        return A, B, C, D, E

    if True:
        # doesn't seem to need pivoting (which is good since it can't have it for BA)
        EAT, BA = scipy.linalg.qr_multiply(
            np.hstack([B, A]),
            np.hstack([E, A]).T,
            pivoting=False,
            conjugate=False,
            mode="right",
        )
        EA = EAT.T
        B = BA[:, :Ninputs]
        E = EA[:, :Nstates]
        # A = EA[:, Nstates:]
        A = BA[:, Ninputs:]
        # del BA, EAT, EA
        # return A, B, C, D, E
        # print(tabulate.tabulate(abs(Ast[:9, :9])))
        # return A, B, C, D, E

    # Ast = Ast[::-1, ::-1]
    print()
    # A = A[::-1, ::-1]
    # E = E[::-1, ::-1]
    # B = B[::-1, :]
    # C = C[:, ::-1]
    # return A, B, C, D, E
    print(Ninputs)
    print(tabulate.tabulate(abs(E[:9, :9])))
    # A = A[::-1, ::-1]
    # E = E[::-1, ::-1]
    # B = B[::-1, :]
    # C = C[:, ::-1]
    Q, ET, P = scipy.linalg.qr(
        E.T,
        pivoting=True,
    )
    E = E[P, :]
    A = A[P, :]
    B = B[P, :]
    E = E @ Q.T
    A = A @ Q.T
    C = C @ Q.T
    # print("Q", tabulate.tabulate(abs(Q[-9:, -9:])))
    # print("A", tabulate.tabulate(abs(A[:9, :9])))
    print("E", tabulate.tabulate(abs(E[:9, :9])))
    return A, B, C, D, E

    # print("SHAPE", A.shape, C.shape)
    # ACT, ET  = scipy.linalg.qr_multiply(
    #    E.T,
    #    np.hstack([A.T, C.T]),
    #    pivoting = False,
    #    conjugate = True,
    #    mode = 'left'
    # )
    # print("1", E.shape, ET.shape)
    # E = ET.T
    # AC = ACT.T
    # print("SHAPE", AC.shape, A.shape, C.shape)
    # A = AC[:Nconstr, :]
    # C = AC[Nconstr:, :]
    # del ET, ACT, AC
    # return A, B, C, D, E


def controllable_staircase(ABCDE, tol=1e-9):
    """
    NOT BROKEN JUST REDUNDANT
    Implementation of
    COMPUTATION  OF IRREDUCIBLE  GENERALIZED STATE-SPACE REALIZATIONS ANDRAS VARGA
    using givens rotations.

    it is very slow, but numerically stable

    TODO, add pivoting,
    TODO, make it use the U-T property on E better for speed
    TODO, make it output Q and Z to apply to aux matrices, perhaps use them on C
    """
    A, B, C, D, E = ABCDE
    # from icecream import ic
    # import tabulate
    Ninputs = B.shape[1]
    Nstates = A.shape[0]
    Nconstr = A.shape[1]
    Noutput = C.shape[0]

    # BA, E = scipy.linalg.qr_multiply(
    #    E,
    #    np.hstack([B, A]),
    #    pivoting = False,
    #    mode = 'left'
    # )

    E, Qapply, P = matrix_algorithms.QR(
        mat=E,
        mshadow=None,
        Qapply=dict(
            A=dict(
                mat=A,
                applyQadj=True,
                applyP=True,
            ),
            B=dict(
                mat=B,
                applyQadj=True,
            ),
            C=dict(
                mat=C,
                applyP=True,
            ),
        ),
        pivoting=True,
        method="Householder",
        # method       = 'Givens',
        overwrite=True,
        Rexact=False,
        # zero_test    = zero_test,
        # select_pivot = True,
    )
    # return A, B, C, D, E

    A = Qapply["A"]
    B = Qapply["B"]
    C = Qapply["C"]
    BA = np.hstack([B, A])

    Nmin = min(Nconstr, Nstates)
    for CidxBA in range(0, Nmin - 1):
        for RidxBA in range(Nconstr - 1, CidxBA, -1):
            # create a givens rotation for Q reduction on BA
            BAv0 = BA[RidxBA - 1, CidxBA]
            BAv1 = BA[RidxBA, CidxBA]
            BAvSq = BAv0 ** 2 + BAv1 ** 2
            if BAvSq < tol:
                continue
            BAvAbs = BAvSq ** 0.5
            c = BAv1 / BAvAbs
            s = BAv0 / BAvAbs
            M = np.array([[s, +c], [-c, s]])
            BA[RidxBA - 1 : RidxBA + 1, :] = M @ BA[RidxBA - 1 : RidxBA + 1, :]

            # TODO, use the U-T to be more efficient
            E[RidxBA - 1 : RidxBA + 1, :] = M @ E[RidxBA - 1 : RidxBA + 1, :]

            Cidx = RidxBA
            Ridx = RidxBA
            CidxZ = RidxBA
            RidxZ = RidxBA

            # row and col swap
            Ev0 = E[Ridx, Cidx - 1]
            Ev1 = E[Ridx, Cidx]
            EvSq = Ev0 ** 2 + Ev1 ** 2
            if EvSq < tol:
                continue
            EvAbs = EvSq ** 0.5
            c = Ev0 / EvAbs
            s = Ev1 / EvAbs
            MT = np.array([[s, +c], [-c, s]])
            # BA[:, Ninputs:][:, Cidx-1:Cidx+1] = BA[:, Ninputs:][:, Cidx-1:Cidx+1] @ MT
            # C[:, Cidx-1:Cidx+1] = C[:, Cidx-1:Cidx+1] @ MT
            # TODO, use the U-T to be more efficient
            # E[:, Cidx-1:Cidx+1] = E[:, Cidx-1:Cidx+1] @ MT
            def ZapplyGRfull(mtrx):
                mtrx[:, CidxZ - 1 : CidxZ + 1] = mtrx[:, CidxZ - 1 : CidxZ + 1] @ MT

            def ZapplyGRfull_shift(mtrx):
                mtrx[:, Ninputs + CidxZ - 1 : Ninputs + CidxZ + 1] = (
                    mtrx[:, Ninputs + CidxZ - 1 : Ninputs + CidxZ + 1] @ MT
                )

            ZapplyGRfull(E)
            ZapplyGRfull(C)
            ZapplyGRfull_shift(BA)

    B = BA[:, :Ninputs]
    A = BA[:, Ninputs:]
    import tabulate

    with open("textmat.txt", "w") as F:
        F.write(tabulate.tabulate(E))
    return A, B, C, D, E
