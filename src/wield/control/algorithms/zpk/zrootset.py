#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import sys
import numbers
from wield.bunch import Bunch
from .roots_matching import nearest_pairs

import numpy as np


def abs_sq(x):
    return x.real ** 2 + x.imag ** 2


class ZDomainRootSet(object):
    def __init__(
        self,
        cplx_plane=None,
        real_line=None,
        disc_line=None,
        p_one_point=None,
        n_one_point=None,
        hermitian=None,
        time_symm=None,
        mirror_real=None,
        mirror_disc=None,
        dt=None,
    ):
        """
        """
        if hermitian is not None:
            if mirror_real is not None:
                assert(mirror_real == hermitian)
            else:
                mirror_real = hermitian
        else:
            hermitian = mirror_real
            assert(mirror_real is not None)

        if time_symm is not None:
            if mirror_disc is not None:
                assert(mirror_disc == time_symm)
            else:
                mirror_disc = time_symm
        else:
            time_symm = mirror_disc
            assert(mirror_disc is not None)

        if real_line is None:
            real_line = np.array([])
        else:
            real_line = np.asarray(real_line)

        if disc_line is None:
            disc_line = np.array([])
        else:
            disc_line = np.asarray(disc_line)

        if cplx_plane is None:
            cplx_plane = np.array([])
        else:
            cplx_plane = np.asarray(cplx_plane)

        if mirror_real:
            assert(real_line is not None)
            cplx_plane = np.asarray(cplx_plane)
            assert(np.all(cplx_plane.imag > 0))
        
        if mirror_disc:
            assert(disc_line is not None)
            assert(np.all(abs(cplx_plane) < 1))
            if mirror_real:
                assert(p_one_point is not None)
                assert(np.all(disc_line.imag > 0))
                assert(np.all(abs(real_line.real) < 1))

        if p_one_point is None:
            p_one_point = np.array([])
        elif isinstance(p_one_point, numbers.Integral):
            p_one_point = np.ones(p_one_point)

        if n_one_point is None:
            n_one_point = np.array([])
        elif isinstance(n_one_point, numbers.Integral):
            n_one_point = -np.ones(n_one_point)

        self.cplx_plane = cplx_plane
        self.real_line = real_line
        self.disc_line = disc_line
        self.p_one_point = p_one_point
        self.n_one_point = n_one_point

        self.mirror_real = mirror_real
        self.hermitian = hermitian
        self.mirror_disc = time_symm
        self.time_symm = time_symm

        return

    def __iter__(self):
        for z in self.p_one_point:
            yield z

        for z in self.n_one_point:
            yield z

        if self.mirror_real:
            if self.mirror_disc:
                for root in self.real_line:
                    yield root
                    yield 1/root
                for root in self.disc_line:
                    yield root
                    yield root.conjugate()
                for root in self.cplx_plane:
                    yield root
                    yield root.conjugate()
                    yield 1/root
                    yield 1/root.conjugate()
            else:
                for root in self.real_line:
                    yield root
                for root in self.disc_line:
                    yield root
                    yield root.conjugate()
                for root in self.cplx_plane:
                    yield root
                    yield root.conjugate()
        else:
            if self.mirror_disc:
                for root in self.real_line:
                    yield root
                    yield 1/root
                for root in self.disc_line:
                    yield root
                for root in self.cplx_plane:
                    yield root
                    yield 1/root
            else:
                for root in self.real_line:
                    yield root
                for root in self.disc_line:
                    yield root
                for root in self.cplx_plane:
                    yield root

    def all(self):
        return np.array(list(self))

    def astuple(self):
        """
        Return self as a tuple of all roots
        """
        return tuple(self)

    def __len__(self):
        length = len(self.p_one_point) + len(self.n_one_point)

        if self.mirror_real:
            if self.mirror_disc:
                length += 2 * len(self.real_line)
                length += 2 * len(self.disc_line)
                length += 4 * len(self.cplx_plane)
            else:
                length += len(self.real_line)
                length += 2 * len(self.disc_line)
                length += 2 * len(self.cplx_plane)
        else:
            if self.mirror_disc:
                length += 2 * len(self.real_line)
                length += len(self.disc_line)
                length += 2 * len(self.cplx_plane)
            else:
                length += len(self.real_line)
                length += len(self.disc_line)
                length += len(self.cplx_plane)
        return length

    def str_iter(self, divide_by=1, real_format_func=None, pos_format_func=None):
        if real_format_func is None:
            def real_format_func(val):
                return "{}".format(val)

            def real_format_func(val):
                return "{: < #22.16g}".format(val)

            if pos_format_func is None:
                def pos_format_func(val):
                    return "{: <#22.16g}".format(val)
        else:

            if pos_format_func is None:
                pos_format_func = real_format_func

        for z in self.p_one_point:
            yield "1"

        for z in self.n_one_point:
            yield "-1"

        if self.mirror_real:
            if self.mirror_disc:
                for root in self.real_line:
                    yield "{}^±1".format(real_format_func(root.real).rstrip())
                for root in self.disc_line:
                    yield "e^(±{}j)".format(pos_format_func(np.angle(root)).strip())
                for root in self.cplx_plane:
                    yield "{}^±1·e^(±{}j)".format(
                        pos_format_func(abs(root)),
                        pos_format_func(np.angle(root)).strip(),
                    )
            else:
                for root in self.real_line:
                    yield "{}".format(real_format_func(root.real).rstrip())
                for root in self.disc_line:
                    yield "e^({}j)".format(real_format_func(np.angle(root)).strip())
                for root in self.cplx_plane:
                    yield "{}^±1·e^({}j)".format(
                        pos_format_func(abs(root)),
                        pos_format_func(np.angle(root)).strip(),
                    )
        elif self.mirror_disc:
            for root in self.real_line:
                yield "{}^±1".format(real_format_func(root.real).rstrip())
            for root in self.disc_line:
                yield "e^({}j)".format(real_format_func(np.angle(root)).strip())
            for root in self.cplx_plane:
                yield "{}^±1·e^({}j)".format(
                    pos_format_func(abs(root)),
                    real_format_func(np.angle(root)).strip(),
                )
        else:
            for root in self.real_line:
                yield "{}".format(real_format_func(root.real).rstrip())
            for root in self.disc_line:
                yield "e^({}j)".format(real_format_func(np.angle(root)).strip())
            for root in self.cplx_plane:
                yield "{}·e^({}j)".format(
                    real_format_func(abs(root)),
                    real_format_func(np.angle(root)).strip(),
                )

    def drop_mirror_real(self):
        if not self.mirror_real:
            return self

        return self.__class__(
            cplx_plane=np.concatenate([self.cplx_plane, self.cplx_plane.conjugate()]),
            real_line=self.real_line,
            disc_line=np.concatenate([self.disc_line, self.disc_line.conjugate()]),
            p_one_point=self.p_one_point,
            n_one_point=self.n_one_point,
            mirror_real=False,
            mirror_disc=self.mirror_disc,
        )

    def flip_to_stable(self):
        cplx_plane = np.copy(self.cplx_plane)
        select = cplx_plane.real > 0
        cplx_plane[select] = -cplx_plane[select].conjugate()
        return self.__class__(
            cplx_plane=cplx_plane,
            real_line=-abs(self.real_line),
            disc_line=self.disc_line,
            p_one_point=self.p_one_point,
            n_one_point=self.n_one_point,
            mirror_real=self.mirror_real,
            mirror_disc=self.mirror_disc,
        )

    def drop_mirror_imag(self):
        if not self.mirror_disc:
            return self

        return self.__class__(
            cplx_plane=np.concatenate([self.cplx_plane, -self.cplx_plane.conjugate()]),
            real_line=np.concatenate([self.real_line, -self.real_line]),
            disc_line=self.disc_line,
            p_one_point=self.p_one_point,
            n_one_point=self.n_one_point,
            mirror_real=self.mirror_real,
            mirror_disc=False,
        )

    def drop_mirror_any(self):
        if not self.mirror_disc and not self.mirror_real:
            return self

        return self.__class__(
            cplx_plane=np.concatenate([
                self.cplx_plane,
                -self.cplx_plane,
                self.cplx_plane.conjugate(),
                -self.cplx_plane.conjugate(),
            ]),
            real_line=np.concatenate([self.real_line, -self.real_line]),
            disc_line=np.concatenate([self.disc_line, -self.disc_line]),
            p_one_point=self.p_one_point,
            n_one_point=self.n_one_point,
            mirror_real=False,
            mirror_disc=False,
        )

    def drop_symmetry(self, mirror_real=None, mirror_disc=None):
        if mirror_real is not None:
            if mirror_disc is not None:
                if not mirror_disc and not mirror_real:
                    return self.drop_mirror_any()
                elif not mirror_disc:
                    return self.drop_mirror_imag()
                else:
                    return self.drop_mirror_real()
            else:
                if not mirror_real:
                    return self.drop_mirror_real()
        elif mirror_disc is not None:
            if not mirror_disc:
                return self.drop_mirror_imag()
        return self

    def __mul__(self, other):
        """
        """
        if isinstance(other, ZDomainRootSet):
            mirror_real = self.mirror_real and other.mirror_real
            mirror_disc = self.mirror_disc and other.mirror_disc
            self = self.drop_symmetry(mirror_real=mirror_real, mirror_disc=mirror_disc)
            other = other.drop_symmetry(mirror_real=mirror_real, mirror_disc=mirror_disc)
            return self.__class__(
                p_one_point=np.concatenate([self.p_one_point, other.p_one_point]),
                n_one_point=np.concatenate([self.n_one_point, other.n_one_point]),
                cplx_plane=np.concatenate([self.cplx_plane, other.cplx_plane]),
                real_line=np.concatenate([self.real_line, other.real_line]),
                disc_line=np.concatenate([self.disc_line, other.disc_line]),
                mirror_disc=mirror_disc,
                mirror_real=mirror_real,
            )
        else:
            return NotImplemented

    def __str__(self):
        return self.normalized_str()

    def normalized_str(
            self,
            divided_by=1,
            real_format_func=None,
    ):
        arr = np.array(list(self.str_iter(divide_by=divided_by, real_format_func=real_format_func)))
        return np.array2string(arr, formatter={'numpystr': str}, separator=', ')

    def fresponse_lnG(self, X, h=1, lnG=0):
        """
        returns the value as if it were generated from a polynomial with last
        coefficient 1 given a coefficient representation and the X_scale.
        """
        X = np.asarray(X)
        h = np.array(h, copy=True, dtype=np.complex128)
        X, h = np.broadcast_arrays(X, h)

        # note that this modifies in-place
        def VfR(roots, h, lnG):
            if len(roots) == 0:
                return h, lnG
            roots = np.asarray(roots)
            mlen = len(roots)
            group_len = 5
            for idx in range((mlen - 1) // group_len + 1):
                r = roots[idx * group_len: (idx + 1) * group_len]
                h = h * np.polynomial.polynomial.polyvalfromroots(X, r)
                abs_max = np.nanmax(abs_sq(h))**0.5
                if abs_max != 0:
                    h /= abs_max
                    lnG += np.log(abs_max)
            return h, lnG

        h, lnG = VfR(self.cplx_plane, h, lnG)
        h, lnG = VfR(self.real_line, h, lnG)
        h, lnG = VfR(self.disc_line, h, lnG)
        h, lnG = VfR(self.p_one_point, h, lnG)
        h, lnG = VfR(self.n_one_point, h, lnG)
        if self.mirror_real:
            h, lnG = VfR(self.cplx_plane.conjugate(), h, lnG)
            h, lnG = VfR(self.disc_line.conjugate(), h, lnG)
        if self.mirror_disc:
            # self.mirror_real not true, self.mirror_disc true
            h, lnG = VfR(1/self.cplx_plane, h, lnG)
            h, lnG = VfR(1/self.real_line, h, lnG)
            if self.mirror_real:
                h, lnG = VfR(1/self.cplx_plane.conjugate(), h, lnG)
        return h, lnG


class RootClassifiers:
    def __init__(
        self,
        line_atol=1e-6,
        line_rtol=1e-6,
        match_atol=1e-6,
        match_rtol=1e-6,
        one_atol=1e-6,
    ):
        def are_same(r1, r2):
            if abs(r1) < 0.8:
                return abs(r1 - r2) < match_atol
            else:
                return abs((r1 / r2) - 1) < match_rtol

        def are_real(r1):
            if abs(r1.real) < 0.8:
                return abs(np.imag(r1)) < line_atol
            else:
                return abs(np.imag(r1) / np.real(r1)) < line_rtol

        def are_unit(r1):
            return abs(abs(np.real(r1))-1) < line_atol

        def are_p_one(r1):
            return abs(r1 - 1) < one_atol

        def are_n_one(r1):
            return abs(r1 + 1) < one_atol

        self.lax_line_tol = 0
        self.line_atol = line_atol
        self.line_rtol = line_rtol
        self.match_atol = match_atol
        self.match_rtol = match_rtol
        self.one_atol = one_atol
        self.are_same = np.vectorize(are_same, otypes=[bool])
        self.are_real = np.vectorize(are_real, otypes=[bool])
        self.are_unit = np.vectorize(are_unit, otypes=[bool])
        self.are_p_one = np.vectorize(are_p_one, otypes=[bool])
        self.are_n_one = np.vectorize(are_n_one, otypes=[bool])

    def R2MR(self, roots_u):
        real_select = self.are_real(roots_u)
        roots_r = roots_u[real_select].real
        roots_u = roots_u[~real_select]

        pos_select = roots_u.imag > 0
        roots_c_neg = roots_u[~pos_select]
        roots_c_pos = roots_u[pos_select]
        rPB = nearest_pairs(roots_c_pos, roots_c_neg.conjugate())
        if self.lax_line_tol > 0:
            roots_u = []
            roots_r2 = []

            def check_ins(u):
                if abs(u.real) > self.lax_line_tol:
                    if abs(u.imag) < self.lax_line_tol:
                        roots_r2.append(u.real)
                    else:
                        roots_u.append(u)
                else:
                    if abs(u.imag / u.real) < self.lax_line_tol:
                        roots_r2.append(u.real)
                    else:
                        roots_u.append(u)

            for u in rPB.l1_remain:
                check_ins(u)
            for u in rPB.l2_remain:
                check_ins(u.conjugate())
            roots_r = np.concatenate([roots_r, roots_r2])
        else:
            roots_u = list(rPB.l1_remain) + [r.conjugate() for r in rPB.l2_remain]
        roots_c = []
        for r1, r2 in rPB.r12_list:
            if self.are_same(r1, r2):
                roots_c.append(r1)
            else:
                roots_u.append(r1)
                roots_u.append(r2.conjugate())
        roots_c = np.array(roots_c)
        roots_u = np.array(roots_u)
        return Bunch(
            c=roots_c,
            r=roots_r,
            u=roots_u,
        )

    def R2MD(self, roots_u, roots_r=[]):
        imag_select = self.are_unit(roots_u)
        roots_d = roots_u[imag_select]
        roots_u = roots_u[~imag_select]

        inner_select = abs(roots_u) < 1
        roots_c_inner = roots_u[inner_select]
        roots_c_outer = roots_u[~inner_select]
        rPB = nearest_pairs(roots_c_inner, 1/roots_c_outer)

        # fill the list with as many pairs as possible
        r12_list_full = rPB.r12_list
        while rPB.r12_list:
            rPB = nearest_pairs(rPB.l1_remain, rPB.l2_remain)
            r12_list_full.extend(rPB.r12_list)
        rPB.r12_list = r12_list_full

        if self.lax_line_tol > 0:
            roots_u = []
            roots_d2 = []

            def check_ins(u):
                if abs(u.imag) > self.lax_line_tol:
                    if abs(u.real) < self.lax_line_tol:
                        roots_d2.append(u.imag)
                    else:
                        roots_u.append(u)
                else:
                    if abs(u.real / u.imag) < self.lax_line_tol:
                        roots_d2.append(u.imag)
                    else:
                        roots_u.append(u)

            for u in rPB.l1_remain:
                check_ins(u)
            for u in rPB.l2_remain:
                check_ins(1/u)
            roots_d = np.concatenate([roots_d, roots_d2])
        else:
            roots_u = list(rPB.l1_remain) + [1/r for r in rPB.l2_remain]

        roots_c = []
        for r1, r2 in rPB.r12_list:
            if self.are_same(r1, r2):
                roots_c.append(r1)
            else:
                roots_u.append(r1)
                roots_u.append(1/r2)

        # this part only meaningfully runs if roots_r has been filled from
        # a previous run of R2MR
        roots_r = np.asarray(roots_r)

        # now mirror over the real line
        select_p_one = self.are_p_one(roots_r)
        roots_r = roots_r[~select_p_one]
        po = np.count_nonzero(select_p_one)
        select_n_one = self.are_n_one(roots_r)
        roots_r = roots_r[~select_n_one]
        no = np.count_nonzero(select_n_one)

        select_outer = abs(roots_r) > 1
        rPB = nearest_pairs(roots_r[select_outer], 1/roots_r[~select_outer])

        # TODO, there is no lax_line_tol for the roots_r mirroring
        roots_u = roots_u + list(rPB.l1_remain) + [1/r for r in rPB.l2_remain]

        roots_r = []
        for r1, r2 in rPB.r12_list:
            if self.are_same(r1, r2):
                roots_r.append(r1)
            else:
                roots_u.append(r1)
                roots_u.append(1/r2)

        # now convert everything

        roots_c = np.array(roots_c)
        roots_u = np.array(roots_u)
        return Bunch(
            c=roots_c,
            r=roots_r,
            d=roots_d,
            u=roots_u,
            po=po,
            no=no,
        )

    def classify_function(
            self,
            tRootSet,
            hermitian,
            time_symm,
    ):

        if hermitian:
            if time_symm:
                def to_rootset(roots, rtype):
                    b = self.R2MR(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    b = self.R2MD(
                        roots_u=b.c,
                        roots_r=b.r,
                    )
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        cplx_plane=b.c,
                        real_line=b.r,
                        disc_line=b.i,
                        p_one_point=b.z,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
            else:
                def to_rootset(roots, rtype):
                    b = self.R2MR(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        cplx_plane=b.c,
                        real_line=b.r,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
        else:
            if time_symm:
                def to_rootset(roots, rtype):
                    b = self.R2MD(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        cplx_plane=b.c,
                        disc_line=b.i,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
            else:
                def to_rootset(roots, rtype):
                    return tRootSet(
                        cplx_plane=roots,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )

        return to_rootset

default_root_classifier = RootClassifiers()
