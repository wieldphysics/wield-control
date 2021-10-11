"""
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy
import scipy.signal
import scipy.linalg

from .eig_algorithms import eigspaces_right_real
from . import matrix_algorithms


def inverse_DSS(A, B, C, D, E):
    constrN = A.shape[0]
    statesN = A.shape[1]
    inputsN = B.shape[1]
    outputN = C.shape[0]

    constrNnew = constrN + inputsN
    statesNnew = statesN + inputsN
    inputsNnew = inputsN
    outputNnew = outputN

    assert(inputsN == outputN)

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

    return newA, newB, newC, newD, newE


def reduce_modal(
        A, B, C, D, E,
        mode = 'C',
        tol = 1e-7,
        use_svd = False
):
    """
    TODO simplify a bunch!
    """
    v_pairs = eigspaces_right_real(A, E, tol = tol)
    A_projections = []
    if mode == 'C':
        #should maybe use SVD for rank estimation?

        #evects are columns are eig-idx, rows are eig-vectors
        for eigs, evects in v_pairs:
            #columns are B-in, rows are SS-eigen
            q, r, P = scipy.linalg.qr((E @ evects).T @ B, pivoting = True)
            for idx in range(q.shape[0] - 1, -1, -1):
                if np.sum(r[idx]**2) < tol:
                    continue
                else:
                    idx += 1
                    break
            idx_split = idx
            A_project = evects @ q[idx_split : ].T
            if A_project.shape[1] > 0:
                A_projections.append(A_project)
    elif mode == 'O':
        #TODO untested so far
        for eigs, evects in v_pairs:
            print(eigs)
            print(evects)
            #columns are C-out, rows are SS-eigen
            q, r, P = scipy.linalg.qr((C @ evects).T, pivoting = True)
            for idx in range(q.shape[0] - 1, -1, -1):
                if np.sum(r[idx]**2) < tol:
                    continue
                else:
                    idx += 1
                    break
            idx_split = idx
            A_project = evects @ q[idx_split : ].T
            if A_project.shape[1] > 0:
                A_projections.append(A_project)
    else:
        raise RuntimeError("Can only reduce mode='C' or 'O'")

    if not A_projections:
        return A, B, C, D, E, False

    A_projections = np.hstack(A_projections)
    Aq, Ar = scipy.linalg.qr(A_projections, mode = 'economic')
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
        p_project_kerU = Au[:, idx_split :]
    else:
        Aq, Ar, Ap = scipy.linalg.qr(A_SS_projection, mode = 'economic', pivoting = True)
        for idx in range(Aq.shape[0] - 1, -1, -1):
            if np.sum(Ar[idx]**2) < tol:
                continue
            else:
                idx += 1
                break
        idx_split = idx
        p_project_imU = Aq[:, :idx_split]
        p_project_kerU = Aq[:, idx_split :]

    E_projections = E @ p_project_kerU
    Eq, Er = scipy.linalg.qr(E_projections, mode = 'economic')
    E_SS_projection = np.diag(np.ones(A.shape[0])) - Eq @ Eq.T.conjugate()

    if use_svd:
        Eu, Es, Ev = np.linalg.svd(E_SS_projection)
        p_project_im = Ev[:idx_split]
        p_project_ker = Ev[idx_split : ]
    else:
        Eq, Er, Ep = scipy.linalg.qr(E_SS_projection, mode = 'economic', pivoting = True)
        p_project_im = Eq.T[:idx_split]
        p_project_ker = Eq.T[idx_split : ]

    #TODO, have this check the mode argument
    if mode == 'C':
        assert(np.all((p_project_ker @ B)**2 < tol))
    if mode == 'O':
        assert(np.all((C @ p_project_kerU)**2 < tol))

    B = p_project_im @ B
    A = p_project_im @ A @ p_project_imU
    E = p_project_im @ E @ p_project_imU
    C = C @ p_project_imU
    return A, B, C, D, E, True


def controllable_staircase(
        ABCDE,
        tol = 1e-14,
        zero_test = lambda x : x == 0,
        reciprocal_system = False,
        do_reduce = False,
        debug_path = None,
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
    #from icecream import ic
    #import tabulate
    Ninputs = B.shape[1]
    Nstates = A.shape[0]
    Nconstr = A.shape[1]
    Noutput = C.shape[0]

    E, Qapply, P = matrix_algorithms.QR(
        mat          = E,
        mshadow      = None,
        Qapply       = dict(
            A = dict(
                mat = A,
                applyQadj= True,
                applyP = True,
            ),
            B = dict(
                mat = B,
                applyQadj = True,
            ),
            C = dict(
                mat = C,
                applyP = True,
            )
        ),
        pivoting     = True,
        method       = 'Householder',
        #method       = 'Givens',
        overwrite    = False,
        Rexact       = False,
        #zero_test    = zero_test,
        #select_pivot = True,
    )
    if debug_path is not None:
        import tabulate
        with open(debug_path('textmatE1.txt'), 'w') as F:
            F.write(tabulate.tabulate(E))

    A = Qapply['A']
    B = Qapply['B']
    C = Qapply['C']
    BA = np.hstack([B, A])

    ###For testing
    #if reciprocal_system:
    #    E, A = A, E
    #    return A, B, C, D, E

    ret = matrix_algorithms.GQR(
        matX = BA,
        matY = E,
        QZapply      = dict(
            C = dict(
                mat = C,
                applyZadj = True,
                applyP    = True,
            )
        ),
        shiftXcol = Ninputs,
        Ncols_end = Nstates,
        #pivoting  = True,
        overwrite = False,
        Rexact    = False,
        NHessenberg = Ninputs,
        #zero_test    = zero_test,
        #select_pivot = True,
    )
    print("DONE", ret.Hessenberg_below)
    BA = ret.matX
    E = ret.matY
    C = ret.QZapply['C']

    B = BA[:, :Ninputs]
    A = BA[:, Ninputs:]

    #broken?!
    if do_reduce:
        Nretain = ret.Hessenberg_below + 1
        B = B[:Nretain, :]
        A = A[:Nretain, :Nretain]
        E = E[:Nretain, :Nretain]
        C = C[:, :Nretain]
    if debug_path is not None:
        import tabulate
        with open(debug_path('textmatBA.txt'), 'w') as F:
            F.write(tabulate.tabulate(BA))
        with open(debug_path('textmatE.txt'), 'w') as F:
            F.write(tabulate.tabulate(E))
    if reciprocal_system:
        E, A = A, E
    return A, B, C, D, E
