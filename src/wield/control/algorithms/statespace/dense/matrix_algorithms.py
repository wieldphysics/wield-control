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
from wield.bunch import Bunch
import scipy.signal
import copy


def norm1DcSq(u):
    N = u.shape[-1]
    return np.dot(
        u.reshape(*u.shape[:-2], 1, N).conjugate(),
        u.reshape(*u.shape[:-2], N, 1),
    )[..., 0, 0]


def norm1DrSq(u):
    N = u.shape[-1]
    return np.dot(
        u.reshape(*u.shape[:-2], 1, N),
        u.reshape(*u.shape[:-2], N, 1),
    )[..., 0, 0]


def QR(
    mat,
    mshadow=None,
    Qapply=dict(),
    pivoting=False,
    method="Householder",
    overwrite=False,
    Rexact=False,
    zero_test=lambda x: x == 0,
    select_pivot=None,
):
    OHAUS = 0
    OGIVE = 1

    if mshadow is not None:
        raise NotImplementedError("The shadow methods are not yet fully funtional")

    if not pivoting:

        def do_pivot(Cidx):
            return

    else:
        if select_pivot is None:

            def select_pivot(mtrx):
                Msum = np.sum(abs(mtrx) ** 2, axis=-2)
                if len(Msum.shape) > 1:
                    Msum = np.amax(Msum, axis=mtrx.shape[:-1])
                return np.argmax(Msum)

        if mshadow:
            pivmat = mshadow
        else:
            pivmat = mat
        pivots = list(range(mat.shape[-1]))

        def do_pivot(Cidx):
            Cidx2 = Cidx + select_pivot(pivmat[Cidx:, Cidx:])
            if Cidx2 == Cidx:
                return
            pivots[Cidx], pivots[Cidx2] = pivots[Cidx2], pivots[Cidx]
            swap_col(mat, Cidx, Cidx2)
            if mshadow is not None:
                swap_col(mshadow, Cidx, Cidx2)
            for name, mdict in Qapply.items():
                if mdict.setdefault("applyP", False):
                    swap_col(mdict["mat"], Cidx, Cidx2)
            return

    method = method.lower()
    if method == "householder":
        otype = OHAUS
    elif method == "givens":
        otype = OGIVE
    else:
        raise RuntimeError("Unrecognized transformation mode")

    if not overwrite:
        mat = np.copy(mat)
        if mshadow is not None:
            mshadow = np.copy(mshadow)
        Qapply = copy.deepcopy(Qapply)

    if otype == OGIVE:
        Nmin = min(mat.shape[-2], mat.shape[-1])
        for Cidx in range(0, Nmin):
            for Ridx in range(mat.shape[0] - 1, Cidx, -1):
                # create a givens rotation for Q reduction on mat
                # from
                # On Computing Givens Rotations Reliably and Efficiently
                f = mat[Ridx - 1, Cidx]
                g = mat[Ridx, Cidx]
                if zero_test(g):
                    c = 1
                    cc = 1
                    s = 0
                    sc = 0
                    r = f
                elif zero_test(f):
                    c = 0
                    cc = 0
                    r = abs(g)
                    sc = g / r
                    s = sc.conjugate()
                else:
                    fa = abs(f)
                    rSQ = fa ** 2 + abs(g) ** 2
                    fsgn = f / fa
                    rr = rSQ ** 0.5
                    c = fa / rr
                    s = fsgn * g.conjugate() / rr
                    r = fsgn * rr
                    sc = s.conjugate()
                    cc = c.conjugate()
                M = np.array(
                    [
                        [c, +s],
                        [-sc, cc],
                    ]
                )
                if Rexact:

                    def applyGR(mtrx):
                        mtrx[Ridx - 1 : Ridx + 1, Cidx:] = (
                            M @ mtrx[Ridx - 1 : Ridx + 1, Cidx:]
                        )

                else:

                    def applyGR(mtrx):
                        mtrx[Ridx - 1 : Ridx + 1, Cidx + 1 :] = (
                            M @ mtrx[Ridx - 1 : Ridx + 1, Cidx + 1 :]
                        )
                        mtrx[Ridx - 1, Cidx] = r
                        mtrx[Ridx, Cidx] = 0

                applyGR(mat)
                # print(u)
                # print(mat[Cidx:, Cidx])
                def applyGRfull(mtrx):
                    mtrx[Ridx - 1 : Ridx + 1, :] = M @ mtrx[Ridx - 1 : Ridx + 1, :]

                def applyGRfullA(mtrx):
                    mtrx[:, Ridx - 1 : Ridx + 1] = (
                        mtrx[:, Ridx - 1 : Ridx + 1] @ M.conjugate().T
                    )

                if mshadow is not None:
                    applyGRfull(mshadow)
                for name, mdict in Qapply.items():
                    if mdict.setdefault("applyQadj", False):
                        applyGRfull(mdict["mat"])
                    if mdict.setdefault("applyQ", False):
                        applyGRfullA(mdict["mat"])
    elif otype == OHAUS:
        Nmin = min(mat.shape[-2], mat.shape[-1])
        for Cidx in range(0, Nmin):
            do_pivot(Cidx)
            # use starts as the x vector, will be modified in place
            u = np.copy(mat[Cidx:, Cidx])
            x0 = u[0]
            xNsq = norm1DcSq(u)
            xN = xNsq ** 0.5
            # TODO, need a better threshold test
            if zero_test(x0):
                x0 = 0
                x0N = 1
                alpha = -xN
            else:
                x0N = abs(x0)
                alpha = -(x0 / x0N) * xN
            u[0] -= alpha
            uNsq = 2 * xN * (xN + (x0 ** 2).real / x0N)
            if zero_test(uNsq):
                continue
            # uNsqtest = norm1DcSq(u)
            # import numpy.testing
            # numpy.testing.assert_almost_equal(uNsqtest, uNsq)
            tau = 2 / uNsq

            N = u.shape[0]
            uc = u.conjugate()
            if Rexact:

                def applyHR(mtrx):
                    mtrx[Cidx:, Cidx:] -= tau * np.dot(
                        u.reshape(N, 1), np.dot(uc.reshape(1, N), mtrx[Cidx:, Cidx:])
                    )

            else:

                def applyHR(mtrx):
                    mtrx[Cidx:, Cidx + 1 :] -= tau * np.dot(
                        u.reshape(N, 1),
                        np.dot(uc.reshape(1, N), mtrx[Cidx:, Cidx + 1 :]),
                    )
                    mtrx[Cidx, Cidx] = alpha
                    mtrx[Cidx + 1 :, Cidx] = 0

            applyHR(mat)
            # print(u)
            # print(mat[Cidx:, Cidx])
            def applyHRfull(mtrx):
                mtrx[Cidx:, :] -= tau * np.dot(
                    u.reshape(N, 1), np.dot(uc.reshape(1, N), mtrx[Cidx:, :])
                )

            if mshadow is not None:
                applyHRfull(mshadow)

            def applyHRfullA(mtrx):
                mtrx[:, Cidx:] -= tau * np.dot(
                    np.dot(mtrx[:, Cidx:], u.reshape(N, 1)), uc.reshape(1, N)
                )

            for name, mdict in Qapply.items():
                if mdict.setdefault("applyQadj", False):
                    applyHRfull(mdict["mat"])
                if mdict.setdefault("applyQ", False):
                    applyHRfullA(mdict["mat"])
    else:
        raise NotImplementedError()

    ret = (mat,)

    if mshadow:
        ret = ret + (mshadow)

    if Qapply:
        rQa = dict()
        for name, mdict in Qapply.items():
            rQa[name] = mdict["mat"]
            if not (mdict["applyQ"] or mdict["applyQadj"] or mdict["applyP"]):
                raise RuntimeError("Must specify one of applyQ, or applyQadj")
        ret = ret + (rQa,)

    if pivoting:
        ret = ret + (pivots,)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def GQR(
    matX,
    matY,
    mshadowX=None,
    mshadowY=None,
    QZapply=dict(),
    # pivoting    = False,
    overwrite=False,
    Rexact=False,
    zero_test=lambda x: x == 0,
    tol=1e-9,
    shiftXcol=0,
    Ncols_end=None,
    Ncols_start=0,
    NHessenberg=None,
):
    """
    Implementation of
    COMPUTATION  OF IRREDUCIBLE  GENERALIZED STATE-SPACE REALIZATIONS ANDRAS VARGA
    using givens rotations.

    It requires matY (E) to be in upper triangular form

    it is very slow, but (mostly) numerically stable

    TODO, add pivoting,
    TODO, make it use the U-T property on E better for speed
    TODO, make it output Q and Z to apply to aux matrices, perhaps use them on C
    """
    if not overwrite:
        matX = np.copy(matX)
        matY = np.copy(matY)
        if mshadowX is not None:
            mshadowX = np.copy(mshadowX)
        if mshadowY is not None:
            mshadowY = np.copy(mshadowY)
        QZapply = copy.deepcopy(QZapply)
    if mshadowX is not None:
        raise NotImplementedError("The shadow methods are not yet fully funtional")
    if mshadowY is not None:
        raise NotImplementedError("The shadow methods are not yet fully funtional")

    for name, mdict in QZapply.items():
        if not (
            mdict.setdefault("applyQ", False)
            or mdict.setdefault("applyQadj", False)
            or mdict.setdefault("applyZ", False)
            or mdict.setdefault("applyZadj", False)
            # or
            # mdict['applyP']
        ):
            raise RuntimeError("Must specify one of applyQ, or applyQadj")

    Nmin = min(matX.shape[-2], matX.shape[-1])
    RidxX_limit = Ncols_start
    if Ncols_end is None:
        Ncols_end = Nmin - 1

    RidxX_limit = Ncols_start
    for CidxX in range(Ncols_start, Ncols_end):
        RidxXfr = matX.shape[-2]
        while RidxXfr > RidxX_limit + 1:
            RidxXfr -= 1
            g = matX[RidxXfr, CidxX]
            if zero_test(g):
                continue

            RidxXto = RidxXfr - 1
            f = matX[RidxXto, CidxX]

            if zero_test(g):
                continue
                c = 1
                cc = 1
                s = 0
                sc = 0
                r = f
            elif zero_test(f):
                c = 0
                cc = 0
                r = abs(g)
                sc = g / r
                s = sc.conjugate()
            else:
                fa = abs(f)
                rSQ = fa ** 2 + abs(g) ** 2
                fsgn = f / fa
                rr = rSQ ** 0.5
                c = fa / rr
                s = fsgn * g.conjugate() / rr
                r = fsgn * rr
                sc = s.conjugate()
                cc = c.conjugate()
                # seems to be really necessary to prevent super weak rotations
                # between a large element and small, should likely just be
                # implemented as part of zero_test
                # if rSQ < tol:
                #    continue
            M = np.array(
                [
                    [c, +s],
                    [-sc, cc],
                ]
            )

            # these indexing schemes assume that RidxXto < RidxXfr (the +1 part)
            Rsl = slice(RidxXto, RidxXfr + 1, RidxXfr - RidxXto)

            def applyGRfull(mtrx):
                mtrx[Rsl, :] = M @ mtrx[Rsl, :]

            def applyGRfullA(mtrx):
                mtrx[:, Rsl] = mtrx[:, Rsl] @ M.conjugate().T

            if Rexact:

                def applyGR(mtrx):
                    mtrx[Rsl, CidxX:] = M @ mtrx[Rsl, CidxX:]

                applyGR = applyGRfull
            else:

                def applyGR(mtrx):
                    mtrx[Rsl, CidxX + 1 :] = M @ mtrx[Rsl, CidxX + 1 :]
                    mtrx[Rsl, CidxX] = r
                    mtrx[RidxXfr, CidxX] = 0

            applyGR(matX)
            if mshadowX is not None:
                applyGR(mshadowX)
            applyGRfull(matY)
            if mshadowY is not None:
                applyGRfull(mshadowY)
            for name, mdict in QZapply.items():
                if mdict["applyQadj"]:
                    applyGRfull(mdict["mat"])
                if mdict["applyQ"]:
                    applyGRfullA(mdict["mat"])

            RCidxYfr = RidxXfr
            RCidxYto = RidxXto

            f = matY[RCidxYfr, RCidxYfr]
            g = matY[RCidxYfr, RCidxYto]
            if zero_test(g):
                continue
                c = 1
                cc = 1
                s = 0
                sc = 0
                r = f
            elif zero_test(f):
                c = 0
                cc = 0
                r = abs(g)
                sc = g / r
                s = sc.conjugate()
            else:
                fa = abs(f)
                rSQ = fa ** 2 + abs(g) ** 2
                fsgn = f / fa
                rr = rSQ ** 0.5
                c = fa / rr
                s = fsgn * g.conjugate() / rr
                r = fsgn * rr
                sc = s.conjugate()
                cc = c.conjugate()
                # seems to be really necessary to prevent super weak rotations
                # between a large element and small, should likely just be
                # implemented as part of zero_test
                # if rSQ < tol:
                #    continue
            M = np.array(
                [
                    [c, +s],
                    [-sc, cc],
                ]
            )
            # these assume RCidxYfr > RCidxYto
            Csl = slice(RCidxYto, RCidxYfr + 1, RCidxYfr - RCidxYto)
            Csl_shift = slice(
                shiftXcol + RCidxYto, shiftXcol + RCidxYfr + 1, RCidxYfr - RCidxYto
            )

            def ZapplyGRfull(mtrx):
                mtrx[:, Csl] = mtrx[:, Csl] @ M

            def ZapplyGRfull_shift(mtrx):
                mtrx[:, Csl_shift] = mtrx[:, Csl_shift] @ M

            if Rexact:

                def ZapplyGR(mtrx):
                    mtrx[: RCidxYfr + 1, Csl] = mtrx[: RCidxYfr + 1, Csl] @ M

                ZapplyGR = ZapplyGRfull
            else:

                def ZapplyGR(mtrx):
                    mtrx[: RCidxYfr + 1, Csl] = mtrx[: RCidxYfr + 1, Csl] @ M
                    mtrx[RCidxYfr, RCidxYfr] = r
                    mtrx[RCidxYfr, RCidxYto] = 0

            ZapplyGR(matY)
            if mshadowY is not None:
                ZapplyGR(mshadowY)
            ZapplyGRfull_shift(matX)
            if mshadowX is not None:
                ZapplyGRfull_shift(mshadowX)

            def ZapplyGRfullA(mtrx):
                mtrx[Csl, :] = M.conjugate().T @ mtrx[Csl, :]

            for name, mdict in QZapply.items():
                if mdict["applyZadj"]:
                    ZapplyGRfull(mdict["mat"])
                if mdict["applyZ"]:
                    ZapplyGRfullA(mdict["mat"])
        # reduce until the matrix has its Hessenberg shifts reduced
        if NHessenberg is None:
            RidxX_limit = CidxX + 1
        else:
            while True:
                val = matX[RidxX_limit, CidxX]
                # print("LIMIT", CidxX, RidxX_limit, val, NHessenberg)
                if abs(val) > tol:
                    break
                RidxX_limit -= 1
                NHessenberg -= 1
            if NHessenberg == 0:
                break
            RidxX_limit += 1

    # Hessenberg_below returns at what column the matrix becomes strictly U-T
    ret = Bunch(
        matX=matX,
        matY=matY,
        Hessenberg_below=CidxX,
    )

    if mshadowX:
        ret.mshadowX = mshadowX

    if mshadowY:
        ret.mshadowY = mshadowY

    if QZapply:
        rQZa = dict()
        for name, mdict in QZapply.items():
            rQZa[name] = mdict["mat"]
        ret.QZapply = rQZa

    return ret


def swap_col(m, Cidx1, Cidx2):
    temp = np.copy(m[:, Cidx2])
    m[:, Cidx2] = m[:, Cidx1]
    m[:, Cidx1] = temp


def swap_row(m, Cidx1, Cidx2):
    temp = np.copy(m[Cidx2, :])
    m[Cidx2, :] = m[Cidx1, :]
    m[Cidx1, :] = temp
