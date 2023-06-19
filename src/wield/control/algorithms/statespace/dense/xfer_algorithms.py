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

import control

from ....ss_bare import ssprint


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


def ss2response_laub(A, B, C, D, E=None, sorz=None, use_blocking=False, blocking_condmax=1e12):
    """
    Use Laub's method to calculate the frequency response. Very fast but in some cases less numerically stable.
    Generally OK if the A/E matrix has been balanced first.

    TODO: Use the bdschur method or slycot mb03rd to further reduce the matrix size. Then enhance the back-substituion to do less work.
    In principle this can reduce the work from N^2 to N at every frequency point. That would be a massive speedup but would need numerical testing.

    """
    sorz = np.asarray(sorz)

    if A.shape[-2:] == (0, 0):
        # print("BCAST", A.shape, D.shape)
        return np.broadcast_to(D, sorz.shape + D.shape[-2:])

    if E is not None:
        if np.all(E == np.eye(E.shape[-1])):
            E = None

    if E is None:
        if not use_blocking:
            A, Z = scipy.linalg.schur(A, output='complex')
            B = Z.transpose().conjugate() @ B
            C = C @ Z
            blk_sizes = None
        else:
            # this MIGHT not always be as numerically stable as desired
            A, Z, blk_sizes = bdschur(A, condmax = blocking_condmax)
            A, Z = scipy.linalg.rsf2csf(A, Z, check_finite=True)
            B = np.linalg.inv(Z) @ B
            # ssprint.print_dense_nonzero_M(A)
            C = C @ Z
            # print("BLOCK SIZES: ", blk_sizes)

        diag = (np.diag(A).reshape(1, -1) - sorz.reshape(-1, 1))

        retval = array_solve_triangular(-A, -diag, B, blk_sizes=blk_sizes)

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


def array_solve_triangular(A, D, b, blk_sizes=None):
    """
    Solve a triangular matrix system.
    A is a (M, M). D is (..., M) are broadcasted diagonals, and b is (M, N)
    """
    # b = np.eye(b.shape[-2], dtype=complex)
    # b = b[:, -2:-1]
    # idx = -2
    # D = D[idx:idx+1]

    # make on big single block
    if blk_sizes is None:
        blk_sizes = [A.shape[-1]]

    # to have access to pop
    blk_sizes = list(blk_sizes)

    bwork = np.broadcast_to(b, D.shape[:-1] + b.shape).copy()

    M = A.shape[-1]

    # print("A", A.shape)
    # print("D", D.shape)
    # print("B", b.shape)
    # print("Bwork", bwork.shape)

    # needed for broadcast magic to preserve the size of .shape after indexing
    # Dv = D.reshape(*D.shape, 1)
    # Av = A.reshape(*A.shape, 1)

    block_top = M

    for idx in range(M - 1, -1, -1):
        if idx < block_top:
            block_len = blk_sizes.pop()
            block_top = idx - block_len + 1

        bwork[..., idx, :] /= D[..., idx:idx+1]
        if block_top < idx:
            bwork[..., block_top:idx, :] -= A[block_top:idx, idx:idx+1] * bwork[..., idx:idx+1, :]

    #Atest = A - np.diag(np.diag(A)) + np.diag(D)

    # Atest = A - np.diag(np.diag(A)) + np.diag(D[idx])
    # bwork2 = np.linalg.inv(Atest) @ b
    # print(abs(Atest @ bwork - b) < 1e-6)
    # print(abs(Atest @ bwork[idx] - Atest @ bwork2) < 1e-6)
    return bwork


###########################################################
## The code below originally from python-control/control/canonical.py
## LICENSE: BSD-3-Clause license 
## It has been modified for testing here
############################################################

def _bdschur_condmax_search(aschur, tschur, condmax):
    """Block-diagonal Schur decomposition search up to condmax

    Iterates mb03rd with different pmax values until:
      - result is non-defective;
      - or condition number of similarity transform is unchanging
        despite large pmax;
      - or condition number of similarity transform is close to condmax.

    Parameters
    ----------
    aschur: (N, N) real ndarray
      Real Schur-form matrix
    tschur: (N, N) real ndarray
      Orthogonal transformation giving aschur from some initial matrix a
    condmax: float
      Maximum condition number of final transformation.  Must be >= 1.

    Returns
    -------
    amodal: (N, N) real ndarray
       block diagonal Schur form
    tmodal: (N, N) real ndarray
       similarity transformation give amodal from aschur
    blksizes: (M,) int ndarray
       Array of Schur block sizes
    eigvals: (N,) real or complex ndarray
       Eigenvalues of amodal (and a, etc.)

    Notes
    -----
    Outputs as for slycot.mb03rd

    aschur, tschur are as returned by scipy.linalg.schur.
    """
    from slycot import mb03rd

    # see notes on RuntimeError below
    pmaxlower = None

    # get lower bound; try condmax ** 0.5 first
    pmaxlower = condmax ** 0.5
    amodal, tmodal, blksizes, eigvals = mb03rd(
        aschur.shape[0], aschur, tschur, sort='N', pmax=pmaxlower)
    # print(eigvals)
    cond = np.linalg.cond(tmodal)
    # print('blksizes: ', blksizes, '{:.1e}'.format(cond))
    if cond <= condmax:
        reslower = amodal, tmodal, blksizes, eigvals
    else:
        pmaxlower = 1.0
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, pmax=pmaxlower)
        cond = np.linalg.cond(tmodal)
        if cond > condmax:
            msg = f"minimum {cond=} > {condmax=}; try increasing condmax"
            raise RuntimeError(msg)

    pmax = pmaxlower

    # phase 1: search for upper bound on pmax
    for i in range(50):
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, sort='C', pmax=pmax)
        cond = np.linalg.cond(tmodal)
        # print('blksizes: ', blksizes, '{:.1e}'.format(cond))
        if cond < condmax:
            pmaxlower = pmax
            reslower = amodal, tmodal, blksizes, eigvals
        else:
            # upper bound found; go to phase 2
            pmaxupper = pmax
            break

        if _bdschur_defective(blksizes, eigvals):
            pmax *= 2
        else:
            return amodal, tmodal, blksizes, eigvals
    else:
        # no upper bound found; return current result
        return reslower

    # phase 2: bisection search
    for i in range(50):
        pmax = (pmaxlower * pmaxupper) ** 0.5
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, pmax=pmax)
        cond = np.linalg.cond(tmodal)

        if cond < condmax:
            if not _bdschur_defective(blksizes, eigvals):
                return amodal, tmodal, blksizes, eigvals
            pmaxlower = pmax
            reslower = amodal, tmodal, blksizes, eigvals
        else:
            pmaxupper = pmax

        if pmaxupper / pmaxlower < _PMAX_SEARCH_TOL:
            # hit search limit
            return reslower
    else:
        raise ValueError('bisection failed to converge; pmaxlower={}, pmaxupper={}'.format(pmaxlower, pmaxupper))


def bdschur(a, condmax=None, sort=None):
    """Block-diagonal Schur decomposition

    Parameters
    ----------
        a : (M, M) array_like
            Real matrix to decompose
        condmax : None or float, optional
            If None (default), use 1/sqrt(eps), which is approximately 1e8
        sort : {None, 'continuous', 'discrete'}
            Block sorting; see below.

    Returns
    -------
        amodal : (M, M) real ndarray
            Block-diagonal Schur decomposition of `a`
        tmodal : (M, M) real ndarray
            Similarity transform relating `a` and `amodal`
        blksizes : (N,) int ndarray
            Array of Schur block sizes

    Notes
    -----
    If `sort` is None, the blocks are not sorted.

    If `sort` is 'continuous', the blocks are sorted according to
    associated eigenvalues.  The ordering is first by real part of
    eigenvalue, in descending order, then by absolute value of
    imaginary part of eigenvalue, also in decreasing order.

    If `sort` is 'discrete', the blocks are sorted as for
    'continuous', but applied to log of eigenvalues
    (i.e., continuous-equivalent eigenvalues).

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> amodal, tmodal, blksizes = ct.bdschur(Gs.A)
    >>> amodal                                                   #doctest: +SKIP
    array([[-2.,  0.],
           [ 0., -1.]])

    """
    if condmax is None:
        condmax = np.finfo(np.float64).eps ** -0.5

    if not (np.isscalar(condmax) and condmax >= 1.0):
        raise ValueError('condmax="{}" must be a scalar >= 1.0'.format(condmax))

    a = np.atleast_2d(a)
    if a.shape[0] == 0 or a.shape[1] == 0:
        return a.copy(), np.eye(a.shape[1], a.shape[0]), np.array([])

    aschur, tschur = scipy.linalg.schur(a)
    amodal, tmodal, blksizes, eigvals = _bdschur_condmax_search(
        aschur, tschur, condmax)

    if sort in ('continuous', 'discrete'):
        idxs = np.cumsum(np.hstack([0, blksizes[:-1]]))
        ev_per_blk = [complex(eigvals[i].real, abs(eigvals[i].imag))
                      for i in idxs]

        if sort == 'discrete':
            ev_per_blk = np.log(ev_per_blk)

        # put most unstable first
        sortidx = np.argsort(ev_per_blk)[::-1]

        # block indices
        blkidxs = [np.arange(i0, i0+ilen)
                   for i0, ilen in zip(idxs, blksizes)]

        # reordered
        permidx = np.hstack([blkidxs[i] for i in sortidx])
        rperm = np.eye(amodal.shape[0])[permidx]

        tmodal = tmodal @ rperm.T
        amodal = rperm @ amodal @ rperm.T
        blksizes = blksizes[sortidx]

    elif sort is None:
        pass

    else:
        raise ValueError('unknown sort value "{}"'.format(sort))

    return amodal, tmodal, blksizes


_IM_ZERO_TOL = np.finfo(np.float64).eps ** 0.5
_PMAX_SEARCH_TOL = 1.001


def _bdschur_defective(blksizes, eigvals):
    """Check  for defective modal decomposition

    Parameters
    ----------
    blksizes: (N,) int ndarray
       size of Schur blocks
    eigvals: (M,) real or complex ndarray
       Eigenvalues

    Returns
    -------
    True iff Schur blocks are defective.

    blksizes, eigvals are the 3rd and 4th results returned by mb03rd.
    """
    if any(blksizes > 2):
        return True

    if all(blksizes == 1):
        return False

    # check eigenvalues associated with blocks of size 2
    init_idxs = np.cumsum(np.hstack([0, blksizes[:-1]]))
    blk_idx2 = blksizes == 2

    im = eigvals[init_idxs[blk_idx2]].imag
    re = eigvals[init_idxs[blk_idx2]].real

    if any(abs(im) < _IM_ZERO_TOL * abs(re)):
        return True

    return False







