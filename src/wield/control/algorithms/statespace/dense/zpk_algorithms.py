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
import collections

from . import xfer_algorithms

from ...zpk import order_reduce

from numpy.polynomial.chebyshev import (
    chebfromroots,
    # chebcompanion,
)

from . import ss_algorithms

TupleABCDE = collections.namedtuple("ABCDE", ('A', 'B', 'C', 'D', 'E'))

pi2 = np.pi * 2


def ss2p(
    A,
    B,
    C,
    D,
    E=None,
    idx_in=None,
    idx_out=None,
    Q_rank_cutoff=1e-15,
    Q_rank_cutoff_unstable=None,
    fmt="scipy",
    allow_MIMO=False,
):
    if not allow_MIMO:
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

        if A.shape[-2:] == (0, 0):
            return np.asarray([]), np.asarray([])

        B = B[:, idx_in: idx_in + 1]
        C = C[idx_out: idx_out + 1, :]
        D = D[idx_out: idx_out + 1, idx_in: idx_in + 1]

    if E is None:
        p = scipy.linalg.eig(A, left=False, right=False)
    else:
        p = scipy.linalg.eig(A, E, left=False, right=False)
        p = np.asarray([_ for _ in p if np.isfinite(_.real)])

    if fmt == "IIRrational":
        p = np.asarray(p) / (2 * np.pi)
    elif fmt == "scipy":
        pass
    else:
        raise RuntimeError("Unrecognized fmt parameter")
    return p


def ss2zp(
    A,
    B,
    C,
    D,
    E=None,
    idx_in=None,
    idx_out=None,
    Q_rank_cutoff=1e-15,
    Q_rank_cutoff_unstable=None,
    fmt="scipy",
    allow_MIMO=False,
):
    if not allow_MIMO:
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

        if A.shape[-2:] == (0, 0):
            return np.asarray([]), np.asarray([])

        B = B[:, idx_in: idx_in + 1]
        C = C[idx_out: idx_out + 1, :]
        D = D[idx_out: idx_out + 1, idx_in: idx_in + 1]

    if E is None:
        p = scipy.linalg.eig(A, left=False, right=False)
    else:
        p = scipy.linalg.eig(A, E, left=False, right=False)
        p = np.asarray([_ for _ in p if np.isfinite(_.real)])
    SS = np.block([[A, B], [C, D]])
    if E is None:
        z = scipy.linalg.eig(
            a=SS,
            b=np.diag(np.concatenate([np.ones(A.shape[0]), np.zeros(1)])),
            left=False,
            right=False,
        )
    else:
        SSE = np.block(
            [
                [E, np.zeros(E.shape[0]).reshape(-1, 1)],
                [np.zeros(E.shape[1]).reshape(1, -1), np.zeros(1).reshape(1, 1)],
            ]
        )
        z = scipy.linalg.eig(a=SS, b=SSE, left=False, right=False)
    z = np.asarray([_ for _ in z if np.isfinite(_.real)])
    if Q_rank_cutoff is not None:
        z, p, k = order_reduce.order_reduce_zpk(
            (z, p, 1),
            reduce_c=True,
            reduce_r=True,
            Q_rank_cutoff=Q_rank_cutoff,
            Q_rank_cutoff_unstable=Q_rank_cutoff_unstable,
        )
    if fmt == "IIRrational":
        z = np.asarray(z) / (2 * np.pi)
        p = np.asarray(p) / (2 * np.pi)
    elif fmt == "scipy":
        pass
    else:
        raise RuntimeError("Unrecognized fmt parameter")
    return z, p


def ss2zpk(
    A,
    B,
    C,
    D,
    E=None,
    idx_in=None,
    idx_out=None,
    Q_rank_cutoff=1e-15,
    Q_rank_cutoff_unstable=None,
    F_match_Hz=None,
    fmt="scipy",
):
    z, p = ss2zp(
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        idx_in=idx_in,
        idx_out=idx_out,
        Q_rank_cutoff=Q_rank_cutoff,
        Q_rank_cutoff_unstable=Q_rank_cutoff_unstable,
        fmt="scipy",
    )
    k = 1

    if F_match_Hz is None:
        # TODO, make this discover the "most isolated" poles to check
        s_match_wHz = None
        for root in p:
            if root.real != 0:
                s_match_wHz = root.imag
        if s_match_wHz is None:
            for root in z:
                if root.real != 0:
                    s_match_wHz = root.imag
    else:
        s_match_wHz = F_match_Hz * np.pi

    tf0 = xfer_algorithms.ss2response(
        A,
        B,
        C,
        D,
        E,
        s=np.asarray(1j * s_match_wHz),
        idx_in=0,
        idx_out=0
    )
    # TODO, avoid using scipy.signal for this
    w, zpk0 = scipy.signal.freqs_zpk(z, p, k, s_match_wHz)
    # print('gain setup', tf0, zpk0)
    k = abs(tf0 / zpk0)

    if fmt == "IIRrational":
        z = np.asarray(z) / (2 * np.pi)
        p = np.asarray(p) / (2 * np.pi)
        k = np.asarray(k) * (2 * np.pi) ** (len(z) - len(p))
    elif fmt == "scipy":
        pass
    else:
        raise RuntimeError("Unrecognized fmt parameter")
    return z, p, k


def chebcompanion2(c1, c2):
    # c is a trimmed copy
    if len(c1) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c1) == 2:
        return np.array([[-c1[0] / c1[1]]]), np.array([1])

    n = len(c1) - 1
    mat = np.zeros((n, n), dtype=c1.dtype)
    scl = np.array([1.0] + [np.sqrt(0.5)] * (n - 1))
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[0] = np.sqrt(0.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c1[:-1] / c1[-1]) * (scl / scl[-1]) * 0.5
    c2x = -(c2[:-1] / c1[-1]) * (scl / scl[-1]) * 0.5
    return mat, c2x


def zpk2cDSS(z, p, k, rescale=None, mode="CCF"):
    z = np.asarray(z)
    p = np.asarray(p)
    if rescale is None:
        rescale = max(np.max(abs(z)), np.max(abs(p))) * 1j
    z = np.asarray(z) * (2 * np.pi)
    p = np.asarray(p) * (2 * np.pi)
    c_k = k
    c_z = chebfromroots(z / rescale)
    # k /= c_z[-1]
    c_k *= c_z[-1]
    c_z = c_z / c_z[-1]
    c_p = chebfromroots(p / rescale)
    c_k /= c_p[-1]
    c_p = c_p / c_p[-1]
    c_z = np.concatenate([c_z, np.zeros(len(c_p) - len(c_z))])

    A, c_zB = chebcompanion2(c_p, c_z)
    A = A * rescale
    B = -c_zB[: len(c_p) - 1].reshape(-1, 1) * c_k
    C = np.concatenate([np.zeros(len(c_p) - 2), np.ones(1)]).reshape(1, -1)
    D = np.array([[c_z[len(c_p) - 1]]])
    E = np.diag(np.ones(len(c_p) - 1))

    if mode == "CCF":
        pass
    elif mode == "OCF":
        A, B, C, D, E = A.T, C.T, B.T, D.T, E.T
    else:
        raise RuntimeError("Unrecognized Mode, must be CCF or OCF")
    return A, B, C, D, E


def DSS_c2r(A, B, C, D, E, with_imag=False):
    A2 = np.block(
        [
            [A.real, -A.imag],
            [A.imag, A.real],
        ]
    )
    B2 = np.block(
        [
            [B.real],
            [B.imag],
        ]
    )
    if with_imag:
        C2 = np.block(
            [
                [C.real + 1j * C.imag, -C.imag + 1j * C.real],
            ]
        )
        D2 = D
    else:
        C2 = np.block(
            [
                [C.real, -C.imag],
            ]
        )
        D2 = D.real
        assert D.imag == 0
    E2 = np.block(
        [
            [E.real, -E.imag],
            [E.imag, E.real],
        ]
    )
    return A2, B2, C2, D2, E2


def zpk2rDSS(z, p, k, **kwargs):
    A, B, C, D, E = zpk2cDSS(z, p, k, **kwargs)
    A, B, C, D, E = DSS_c2r(A, B, C, D, E)
    reduced = True
    while reduced:
        A, B, C, D, E, reduced = ss_algorithms.reduce_modal(A, B, C, D, E, mode="O")
        if not reduced:
            break
        A, B, C, D, E, reduced = ss_algorithms.reduce_modal(A, B, C, D, E, mode="C")
    return A, B, C, D, E


def poly2ss(
        num,
        den,
        rescale_has=None,
        rescale_do=None,
        mode="CCF"
):
    """
    """
    if rescale_do is not None:
        assert(abs(rescale_do) > 1e-15)
        rescale_arr = rescale_do**(np.arange(len(den)))
        c_p = np.asarray(den) * rescale_arr
        c_z = np.asarray(num) * rescale_arr[: len(num)]
    else:
        c_z = np.asarray(num)
        c_p = np.asarray(den)
        rescale_do = 1

    if rescale_has is not None:
        rescale_do *= rescale_has

    c_k = c_p[-1]
    c_p = c_p / c_k
    c_z = np.concatenate([c_z, np.zeros(len(c_p) - len(c_z))])

    K = len(c_p)
    A = rescale_do * np.block([[np.eye(K - 1, K - 2, -1), -c_p[:-1].reshape(-1, 1)]])
    B = rescale_do * (c_z[: len(c_p) - 1] - (c_z[-1] * c_p[:-1])).reshape(-1, 1) / c_k
    C = np.concatenate([np.zeros(len(c_p) - 2), np.ones(1)]).reshape(1, -1)
    D = np.array([[c_z[len(c_p) - 1]]]) / c_k
    E = np.diag(np.ones(len(c_p) - 1))

    if mode == "CCF":
        pass
    elif mode == "OCF":
        A, B, C, D, E = A.T, C.T, B.T, D.T, E.T
    else:
        raise RuntimeError("Unrecognized Mode, must be CCF or OCF")
    return A, B, C, D, E


def zpk_cascade(zr, zc, pr, pc, k, convention="scipy"):
    return zpkdict_cascade(
        zdict=dict(r=zr, c=zc),
        pdict=dict(r=pr, c=pc),
        k=k,
        convention=convention,
    )


def zpkdict_cascade(
    zdict,
    pdict,
    k,
    convention="scipy",
):
    """
    Create a statespace from a cascade of ZPKs.
    """
    if convention is not "scipy":
        raise RuntimeError("Only scipy convention currently supported")

    if True:
        def check(ABCDE):
            for m in ABCDE:
                assert(np.all(np.isfinite(m)))
    else:
        def check(ABCDE):
            return

    def gen_polys(rdict):
        Rc = rdict["c"]
        Rr = rdict["r"]

        poly = []
        for c in Rc:
            poly.append((c.real * c.real + c.imag * c.imag, -2 * c.real, 1))
        idx = 0
        while idx <= len(Rr)  - 2:
            r1, r2 = Rr[idx: idx + 2]
            poly.append((r1 * r2, -(r1 + r2), 1))
            idx += 2
        if idx < len(Rr):
            (r1,) = Rr[idx:]
            last = (-r1, 1)
        else:
            last = None
        return poly, last

    Zpoly, Zlast = gen_polys(zdict)
    Ppoly, Plast = gen_polys(pdict)

    ABCDEs = []
    idx = -1
    for idx in range(min(len(Zpoly), len(Ppoly))):
        zp = Zpoly[idx]
        pp = Ppoly[idx]
        rescale = abs(zp[0] * pp[0])**0.5
        if abs(rescale) < 1e-6:
            rescale = None
        ABCDE = poly2ss(zp, pp, rescale_do=rescale)
        # ABCD = scipy.signal.tf2ss(zp[::-1], pp[::-1])
        # E = np.eye(2)
        check(ABCDE)
        ABCDEs.append(ABCDE)
    if len(Zpoly) <= len(Ppoly):
        for idx in range(len(Zpoly), len(Ppoly)):
            pp = Ppoly[idx]
            rescale = 1 / pp[-1]
            if Zlast is not None:
                zp = Zlast
                Zlast = None
            else:
                zp = [1]
            ABCDE = poly2ss(zp, pp, rescale_do=rescale)
            check(ABCDE)
            ABCDEs.append(ABCDE)
    else:
        for idx in range(len(Ppoly), len(Zpoly)):
            zp = Zpoly[idx]
            rescale = 1 / zp[-1]
            if Plast is not None:
                pp = Plast
                Plast = None
            else:
                pp = [1]
            ABCDE = poly2ss(pp, zp, rescale_do=rescale)
            ABCDE = ss_algorithms.inverse_DSS(*ABCDE)
            check(ABCDE)
            ABCDEs.append(ABCDE)
    if Plast is None:
        if Zlast is not None:
            ABCDE = poly2ss([1], Zlast)
            ABCDE = ss_algorithms.inverse_DSS(*ABCDE)
            check(ABCDE)
            ABCDEs.append(ABCDE)
    else:
        idx += 1
        if Zlast is None:
            Zlast = [1]
        ABCDE = poly2ss(Zlast, Plast)
        check(ABCDE)
        ABCDEs.append(ABCDE)

    if ABCDEs:
        A, B, C, D, E = ABCDEs[0]
        B *= k
        D *= k
        ABCDEs[0] = A, B, C, D, E
    else:
        ABCDE = (
            np.array([[]]).reshape(0, 0),
            np.array([[]]).reshape(0, 1),
            np.array([[]]).reshape(1, 0),
            np.array([[k]]),
            np.array([[]]).reshape(0, 0),
        )
        ABCDEs.append(ABCDE)
    return ABCDEs


def zpk_rc(
    Zc=[],
    Zr=[],
    Pc=[],
    Pr=[],
    k=1,
    convention="scipy",
    orientation="lower",
):
    if convention == "scipyHz":
        convention = "scipy"
        Zc = pi2 * np.array(Zc)
        Zr = pi2 * np.array(Zr)
        Pc = pi2 * np.array(Pc)
        Pr = pi2 * np.array(Pr)
    elif convention == 'scipy':
        pass
    else:
        raise RuntimeError("Unrecognized Convention")

    return ZPKdict(
        zdict=dict(c=Zc, r=Zr),
        pdict=dict(c=Pc, r=Pr),
        k=k,
        convention=convention,
        orientation=orientation,
    )


def ZPKdict(
    zdict,
    pdict,
    k,
    convention="scipy",
    orientation="lower",
):
    """
    Create a statespace from dictionaries of poles and zeros separated into real and imaginary parts
    orientation: Whether the A matrix is "upper" or "lower" triangular real Schur form
    """
    ABCDEs = zpkdict_cascade(
        zdict=zdict, pdict=pdict, k=k, convention=convention
    )
    ABCDE = ss_algorithms.chain(ABCDEs)

    orientation = orientation.lower()
    if orientation == 'lower':
        pass
    elif orientation == 'upper':
        A, B, C, D, E = ABCDE
        ABCDE = (
            A[..., ::-1, ::-1],
            B[..., ::-1, :],
            C[..., :, ::-1],
            D,
            E[..., ::-1, ::-1],
        )
    else:
        raise RuntimeError("Unrecognized Orientation")

    return TupleABCDE(*ABCDE)


