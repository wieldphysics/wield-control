#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import collections
import numpy as np
import scipy
import scipy.signal
import scipy.linalg
from wield.bunch import Bunch

from .eig_algorithms import eigspaces_right_real
from . import matrix_algorithms


TupleABCDE = collections.namedtuple("ABCDE", ('A', 'B', 'C', 'D', 'E'))


def inverse_DSS(A, B, C, D, E):
    constrN = A.shape[0]
    statesN = A.shape[1]
    inputsN = B.shape[1]
    outputN = C.shape[0]

    constrNnew = constrN + inputsN
    statesNnew = statesN + inputsN
    inputsNnew = inputsN
    outputNnew = outputN

    assert inputsN == outputN

    newA = np.zeros((constrNnew, statesNnew))
    newE = np.zeros((constrNnew, statesNnew))
    newB = np.zeros((constrNnew, inputsNnew))
    newC = np.zeros((outputNnew, statesNnew))
    newD = np.zeros((outputNnew, inputsNnew))

    newA[:constrN, :statesN] = A
    newA[:constrN, statesN:] = B
    newA[constrN:, :statesN] = C
    newA[constrN:, statesN:] = D
    newE[:constrN, :statesN] = E
    newB[constrN:, :inputsN] = -1
    newC[:outputN, statesN:] = 1

    return TupleABCDE(newA, newB, newC, newD, newE)


def reduce_modal(A, B, C, D, E, mode="C", tol=1e-7, use_svd=False):
    """
    TODO simplify a bunch!
    """
    v_pairs = eigspaces_right_real(A, E, tol=tol)
    A_projections = []
    if mode == "C":
        # should maybe use SVD for rank estimation?

        # evects are columns are eig-idx, rows are eig-vectors
        for eigs, evects in v_pairs:
            # columns are B-in, rows are SS-eigen
            q, r, P = scipy.linalg.qr((E @ evects).T @ B, pivoting=True)
            for idx in range(q.shape[0] - 1, -1, -1):
                if np.sum(r[idx] ** 2) < tol:
                    continue
                else:
                    idx += 1
                    break
            idx_split = idx
            A_project = evects @ q[idx_split:].T
            if A_project.shape[1] > 0:
                A_projections.append(A_project)
    elif mode == "O":
        # TODO untested so far
        for eigs, evects in v_pairs:
            print(eigs)
            print(evects)
            # columns are C-out, rows are SS-eigen
            q, r, P = scipy.linalg.qr((C @ evects).T, pivoting=True)
            for idx in range(q.shape[0] - 1, -1, -1):
                if np.sum(r[idx] ** 2) < tol:
                    continue
                else:
                    idx += 1
                    break
            idx_split = idx
            A_project = evects @ q[idx_split:].T
            if A_project.shape[1] > 0:
                A_projections.append(A_project)
    else:
        raise RuntimeError("Can only reduce mode='C' or 'O'")

    if not A_projections:
        return A, B, C, D, E, False

    A_projections = np.hstack(A_projections)
    Aq, Ar = scipy.linalg.qr(A_projections, mode="economic")
    A_SS_projection = np.diag(np.ones(A.shape[0])) - Aq @ Aq.T.conjugate()

    if use_svd:
        Au, As, Av = np.linalg.svd(A_SS_projection)
        for idx in range(len(As) - 1, -1, -1):
            if As[idx] < tol:
                continue
            else:
                idx += 1
                break
        idx_split = idx
        p_project_imU = Au[:, :idx_split]
        p_project_kerU = Au[:, idx_split:]
    else:
        Aq, Ar, Ap = scipy.linalg.qr(A_SS_projection, mode="economic", pivoting=True)
        for idx in range(Aq.shape[0] - 1, -1, -1):
            if np.sum(Ar[idx] ** 2) < tol:
                continue
            else:
                idx += 1
                break
        idx_split = idx
        p_project_imU = Aq[:, :idx_split]
        p_project_kerU = Aq[:, idx_split:]

    E_projections = E @ p_project_kerU
    Eq, Er = scipy.linalg.qr(E_projections, mode="economic")
    E_SS_projection = np.diag(np.ones(A.shape[0])) - Eq @ Eq.T.conjugate()

    if use_svd:
        Eu, Es, Ev = np.linalg.svd(E_SS_projection)
        p_project_im = Ev[:idx_split]
        p_project_ker = Ev[idx_split:]
    else:
        Eq, Er, Ep = scipy.linalg.qr(E_SS_projection, mode="economic", pivoting=True)
        p_project_im = Eq.T[:idx_split]
        p_project_ker = Eq.T[idx_split:]

    # TODO, have this check the mode argument
    if mode == "C":
        assert np.all((p_project_ker @ B) ** 2 < tol)
    if mode == "O":
        assert np.all((C @ p_project_kerU) ** 2 < tol)

    B = p_project_im @ B
    A = p_project_im @ A @ p_project_imU
    E = p_project_im @ E @ p_project_imU
    C = C @ p_project_imU
    return A, B, C, D, E, True


def controllable_staircase(
    ABCDE,
    tol=1e-14,
    zero_test=lambda x: x == 0,
    reciprocal_system=False,
    do_reduce=False,
    debug_path=None,
):
    """
    Implementation of
    COMPUTATION  OF IRREDUCIBLE  GENERALIZED STATE-SPACE REALIZATIONS ANDRAS VARGA
    using givens rotations.

    it is very slow, but numerically stable

    TODO, add pivoting,
    TODO, make it use the U-T property on E better for speed
    TODO, make it output Q and Z to apply to aux matrices, perhaps use them on C
    """
    A, B, C, D, E = ABCDE
    if reciprocal_system:
        E, A = A, E
    # from icecream import ic
    # import tabulate
    Ninputs = B.shape[1]
    Nstates = A.shape[0]
    Nconstr = A.shape[1]
    Noutput = C.shape[0]

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
        overwrite=False,
        Rexact=False,
        # zero_test    = zero_test,
        # select_pivot = True,
    )
    if debug_path is not None:
        import tabulate

        with open(debug_path("textmatE1.txt"), "w") as F:
            F.write(tabulate.tabulate(E))

    A = Qapply["A"]
    B = Qapply["B"]
    C = Qapply["C"]
    BA = np.hstack([B, A])

    ###For testing
    # if reciprocal_system:
    #    E, A = A, E
    #    return A, B, C, D, E

    ret = matrix_algorithms.GQR(
        matX=BA,
        matY=E,
        QZapply=dict(
            C=dict(
                mat=C,
                applyZadj=True,
                applyP=True,
            )
        ),
        shiftXcol=Ninputs,
        Ncols_end=Nstates,
        # pivoting  = True,
        overwrite=False,
        Rexact=False,
        NHessenberg=Ninputs,
        # zero_test    = zero_test,
        # select_pivot = True,
    )
    print("DONE", ret.Hessenberg_below)
    BA = ret.matX
    E = ret.matY
    C = ret.QZapply["C"]

    B = BA[:, :Ninputs]
    A = BA[:, Ninputs:]

    # broken?!
    if do_reduce:
        Nretain = ret.Hessenberg_below + 1
        B = B[:Nretain, :]
        A = A[:Nretain, :Nretain]
        E = E[:Nretain, :Nretain]
        C = C[:, :Nretain]
    if debug_path is not None:
        import tabulate

        with open(debug_path("textmatBA.txt"), "w") as F:
            F.write(tabulate.tabulate(BA))
        with open(debug_path("textmatE.txt"), "w") as F:
            F.write(tabulate.tabulate(E))
    if reciprocal_system:
        E, A = A, E
    return A, B, C, D, E


def chain(SSs, orientation='lower'):
    """
    Construct a sequential product of state spaces. Each state space must be a tuple of ABCD or ABCDE
    """
    orientation = orientation.lower()
    ss_seq = []
    constrN = 0
    statesN = 0
    inputsN = 0
    outputN = 0
    for idx, ss in enumerate(SSs):
        ssB = Bunch()
        if len(ss) == 4:
            A, B, C, D = ss
            E = np.eye(A.shape[-1])
            assert(A.shape[-1] == A.shape[-2])
        elif len(ss) == 5:
            A, B, C, D, E = ss
            if E is None:
                E = np.eye(A.shape[-1])

        if orientation == 'upper':
            A, B, C, D, E = (
                A[..., ::-1, ::-1],
                B[..., ::-1, :],
                C[..., :, ::-1],
                D,
                E[..., ::-1, ::-1],
            )
        ssB.A = A
        ssB.B = B
        ssB.C = C
        ssB.D = D
        ssB.E = E
        ssB.sN = slice(statesN, statesN + A.shape[-2])
        ssB.cN = slice(constrN, constrN + A.shape[-1])
        ssB.iN = slice(0, B.shape[-1])
        ssB.oN = slice(0, C.shape[-2])

        if E is not None:
            assert(E.shape == A.shape)

        constrN += A.shape[-2]
        statesN += A.shape[-1]
        if idx == 0:
            inputsN = B.shape[-1]
        else:
            assert(outputN == B.shape[-1])
        # make sure the current output is matched to the previous input
        outputN = C.shape[-2]
        ss_seq.append(ssB)
    del ss

    A = np.zeros((constrN, statesN))
    E = np.zeros((constrN, statesN))
    B = np.zeros((constrN, inputsN))
    C = np.zeros((outputN, statesN))

    # set up the initial array on the first statespace in the sequence
    ssB = ss_seq[0]
    A[..., ssB.cN, ssB.sN] = ssB.A
    E[..., ssB.cN, ssB.sN] = ssB.E
    B[ssB.cN, :] = ssB.B
    D = ssB.D
    for idx_ss, ssB in enumerate(ss_seq[1:], 1):
        A[..., ssB.cN, ssB.sN] = ssB.A
        E[..., ssB.cN, ssB.sN] = ssB.E
        B_ud = ssB.B

        for idx_down in range(idx_ss - 1, -1, -1):
            ss_down = ss_seq[idx_down]
            A[ssB.cN, ss_down.sN] = B_ud @ ss_down.C
            if idx_down == 0:
                break
            B_ud = B_ud @ ss_down.D

        B[ssB.cN, :] = ssB.B @ D
        D = ssB.D @ D

    ssB = ss_seq[-1]
    C[:, ssB.sN] = ssB.C
    D_rev = ssB.D
    # now loop down through the D matrices
    for ssB in ss_seq[-2::-1]:
        C_into = D_rev @ ssB.C
        C[:, ssB.sN] = C_into
        D_rev = D_rev @ ssB.D

    if orientation == 'lower':
        pass
    elif orientation == 'upper':
        A, B, C, D, E = (
            A[..., ::-1, ::-1],
            B[..., ::-1, :],
            C[..., :, ::-1],
            D,
            E[..., ::-1, ::-1],
        )
    else:
        raise RuntimeError("Unrecognized Orientation")

    return TupleABCDE(A, B, C, D, E)


def chainE(SSs, orientation='lower'):
    orientation = orientation.lower()
    ss_seq = []
    constrN = 0
    statesN = 0
    for idx, ss in enumerate(SSs):
        ssB = Bunch()
        if len(ss) == 4:
            A, B, C, D = ss
            E = np.eye(A.shape[-1])
            assert (A.shape[-1] == A.shape[-2])
        elif len(ss) == 5:
            A, B, C, D, E = ss
            if E is None:
                E = np.eye(A.shape[-1])
        if orientation == 'upper':
            A, B, C, D, E = (
                A[..., ::-1, ::-1],
                B[..., ::-1, :],
                C[..., :, ::-1],
                D,
                E[..., ::-1, ::-1],
            )
        ssB.A = A
        ssB.B = B
        ssB.C = C
        ssB.D = D
        ssB.E = E
        ssB.sN = slice(statesN, statesN + A.shape[-2])
        ssB.cN = slice(constrN, constrN + A.shape[-1])
        ssB.iN = slice(0, B.shape[-1])
        ssB.oN = slice(0, C.shape[-2])

        if E is not None:
            assert (E.shape == A.shape)

        ssB.inputsN = B.shape[-1]
        ssB.outputN = C.shape[-2]
        # make sure the current output is matched to the previous input
        if idx != 0:
            assert (ss_seq[idx-1].outputN == ssB.inputsN)

        ssB.sNE = slice(statesN + A.shape[-2], statesN + A.shape[-2] + ssB.outputN)
        ssB.cNE = slice(constrN + A.shape[-1], constrN + A.shape[-1] + ssB.outputN)

        constrN += A.shape[-2]
        statesN += A.shape[-1]
        if idx < len(SSs) - 1:
            # add states and constraints for E chain
            constrN += ssB.outputN
            statesN += ssB.outputN

        ss_seq.append(ssB)
    del ss

    if len(SSs) == 1:
        ssB = ss_seq[0]
        return ssB.A, ssB.B, ssB.C, ssB.D, ssB.E

    A = np.zeros((constrN, statesN))
    E = np.zeros((constrN, statesN))
    B = np.zeros((constrN, ss_seq[0].inputsN))
    C = np.zeros((ss_seq[-1].outputN, statesN))
    D = np.zeros((ss_seq[-1].outputN, ss_seq[0].inputsN))

    ssB = ss_seq[0]
    A[..., ssB.cN, ssB.sN] = ssB.A
    E[..., ssB.cN, ssB.sN] = ssB.E
    B[ssB.cN, :] = ssB.B
    B[ssB.cNE, :] = ssB.D

    A[..., ssB.cNE, ssB.sNE] = -np.eye(ssB.outputN)
    A[..., ssB.cNE, ssB.sN] = ssB.C
    ssBp = ssB

    for idx_ss, ssB in enumerate(ss_seq[1:-1], 1):
        A[..., ssB.cN, ssB.sN] = ssB.A
        E[..., ssB.cN, ssB.sN] = ssB.E

        A[..., ssB.cN, ssBp.sNE] = ssB.B
        A[..., ssB.cNE, ssB.sNE] = -np.eye(ssB.outputN)
        A[..., ssB.cNE, ssBp.sNE] = ssB.D
        A[..., ssB.cNE, ssB.sN] = ssB.C

        ssBp = ssB

    ssB = ss_seq[-1]
    A[..., ssB.cN, ssB.sN] = ssB.A
    E[..., ssB.cN, ssB.sN] = ssB.E
    A[..., ssB.cN, ssBp.sNE] = ssB.B
    C[:, ssB.sN] = ssB.C
    C[:, ssBp.sNE] = ssB.D

    if orientation == 'lower':
        pass
    elif orientation == 'upper':
        A, B, C, D, E = (
            A[..., ::-1, ::-1],
            B[..., ::-1, :],
            C[..., :, ::-1],
            D,
            E[..., ::-1, ::-1],
        )
    else:
        raise RuntimeError("Unrecognized Orientation")

    return TupleABCDE(A, B, C, D, E)
