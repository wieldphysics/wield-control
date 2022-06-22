#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import sys
import numbers
from wavestate.bunch import Bunch
from .roots_matching import nearest_pairs

import numpy as np


def abs_sq(x):
    return x.real ** 2 + x.imag ** 2


class SDomainRootSet(object):
    def __init__(
        self,
        c_plane=None,
        r_line=None,
        i_line=None,
        z_point=None,
        hermitian=None,
        time_symm=None,
        mirror_real=None,
        mirror_imag=None,
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
            if mirror_imag is not None:
                assert(mirror_imag == time_symm)
            else:
                mirror_imag = time_symm
        else:
            time_symm = mirror_imag
            assert(mirror_imag is not None)

        if mirror_real:
            assert(r_line is not None)
            assert(np.all(c_plane.imag > 0))
        
        if mirror_imag:
            assert(self.i_line is not None)
            assert(np.all(c_plane.real < 0))
            if mirror_real:
                assert(self.z_point is not None)
                assert(np.all(i_line.real > 0))
                assert(np.all(r_line.real < 0))

        if z_point is None:
            z_point = np.array([])
        if r_line is None:
            r_line = np.array([])
        if i_line is None:
            i_line = np.array([])
        if c_plane is None:
            c_plane = np.array([])

        self.c_plane = c_plane
        self.r_line = r_line
        self.i_line = i_line
        self.z_point = z_point

        self.mirror_real = mirror_real
        self.hermitian = hermitian
        self.mirror_imag = time_symm
        self.time_symm = time_symm

        if isinstance(z_point, numbers.Integral):
            z_point = np.zeros(z_point)
        return

    def __iter__(self):
        for z in self.z_point:
            yield z

        if self.mirror_real:
            if self.mirror_imag:
                for root in self.r_line:
                    yield root
                    yield -root
                for root in self.i_line:
                    yield root
                    yield -root
                for root in self.c_plane:
                    yield root
                    yield root.conjugate()
                    yield -root
                    yield -root.conjugate()
            else:
                for root in self.r_line:
                    yield root
                for root in self.i_line:
                    yield root
                    yield -root
                for root in self.c_plane:
                    yield root
                    yield root.conjugate()
        else:
            if self.mirror_imag:
                for root in self.r_line:
                    yield root
                    yield root.conjugate()
                for root in self.i_line:
                    yield root
                for root in self.c_plane:
                    yield root
                    yield -root.conjugate()
            else:
                for root in self.r_line:
                    yield root
                for root in self.i_line:
                    yield root
                for root in self.c_plane:
                    yield root

    def astuple(self):
        """
        Return self as a tuple of all roots
        """
        return tuple(self)

    def __len__(self):
        length = len(self.z_point)

        if self.mirror_real:
            if self.mirror_imag:
                length += 2 * len(self.r_line)
                length += 2 * len(self.i_line)
                length += 4 * len(self.c_plane)
            else:
                length += len(self.r_line)
                length += 2 * len(self.i_line)
                length += 2 * len(self.c_plane)
        else:
            if self.mirror_imag:
                length += 2 * len(self.r_line)
                length += len(self.i_line)
                length += 2 * len(self.c_plane)
            else:
                length += len(self.r_line)
                length += len(self.i_line)
                length += len(self.c_plane)
        return length

    def str_iter(self, divide_by=1, real_format_func=None):
        if real_format_func is None:
            def real_format_func(val):
                return "{}".format(val)

        for z in self.z_point:
            yield "0"

        if self.mirror_real:
            if self.mirror_imag:
                for root in self.r_line:
                    yield "±{}".format(real_format_func(-root.real))
                for root in self.i_line:
                    yield "±{}j".format(real_format_func(root.imag))
                for root in self.c_plane:
                    yield "±{}±{}j".format(
                        real_format_func(-root.real),
                        real_format_func(root.imag),
                    )
            else:
                for root in self.r_line:
                    yield "{}".format(real_format_func(root.real))
                for root in self.i_line:
                    yield "±{}j".format(real_format_func(root.imag))
                for root in self.c_plane:
                    yield "{}±{}j".format(
                        real_format_func(root.real),
                        real_format_func(root.imag),
                    )
        elif self.mirror_imag:
            for root in self.r_line:
                yield "±{}".format(real_format_func(-root.real))
            for root in self.i_line:
                yield "{}j".format(real_format_func(root.imag))
            for root in self.c_plane:
                if root.imag > 0:
                    yield "±({}+{}j)".format(
                        real_format_func(-root.real),
                        real_format_func(root.imag),
                    )
                else:
                    yield "±({}-{}j)".format(
                        real_format_func(-root.real),
                        real_format_func(-root.imag),
                    )
        else:
            for root in self.r_line:
                yield "{}".format(real_format_func(root.real))
            for root in self.i_line:
                yield "{}j".format(real_format_func(root.imag))
            for root in self.c_plane:
                if root.imag > 0:
                    yield "{}+{}j".format(
                        real_format_func(-root.real),
                        real_format_func(root.imag),
                    )
                else:
                    yield "{}-{}j".format(
                        real_format_func(-root.real),
                        real_format_func(-root.imag),
                    )

    def drop_mirror_real(self):
        if not self.mirror_real:
            return self

        return self.__class__(
            c_plane=np.concatenate([self.c_plane, -self.c_plane.conjugate()]),
            r_line=self.r_line,
            i_line=np.concatenate([self.i_line, -self.i_line]),
            z_point=self.z_point,
            mirror_real=False,
            mirror_imag=self.mirror_imag,
        )

    def drop_mirror_imag(self):
        if not self.mirror_imag:
            return self

        return self.__class__(
            c_plane=np.concatenate([self.c_plane, self.c_plane.conjugate()]),
            r_line=np.concatenate([self.r_line, -self.r_line]),
            i_line=self.i_line,
            z_point=self.z_point,
            mirror_real=self.mirror_real,
            mirror_imag=False,
        )

    def drop_mirror_any(self):
        if not self.mirror_imag and not self.mirror_real:
            return self

        return self.__class__(
            c_plane=np.concatenate([
                self.c_plane,
                -self.c_plane,
                self.c_plane.conjugate(),
                -self.c_plane.conjugate(),
            ]),
            r_line=np.concatenate([self.r_line, -self.r_line]),
            i_line=np.concatenate([self.i_line, -self.i_line]),
            z_point=self.z_point,
            mirror_real=False,
            mirror_imag=False,
        )

    def drop_symmetry(self, mirror_real=None, mirror_imag=None):
        if mirror_real is not None:
            if mirror_imag is not None:
                if not mirror_imag and not mirror_real:
                    return self.drop_mirror_any()
                elif not mirror_imag:
                    return self.drop_mirror_imag()
                else:
                    return self.drop_mirror_real()
            else:
                if not mirror_real:
                    return self.drop_mirror_real()
        elif mirror_imag is not None:
            if not mirror_imag:
                return self.drop_mirror_imag()
        return self

    def __mul__(self, other):
        """
        """
        if isinstance(other, SDomainRootSet):
            mirror_real = self.mirror_real and other.mirror_real
            mirror_imag = self.mirror_imag and other.mirror_imag
            self = self.drop_symmetry(mirror_real=mirror_real, mirror_imag=mirror_imag)
            other = other.drop_symmetry(mirror_real=mirror_real, mirror_imag=mirror_imag)
            return self.__class__(
                z_point=np.concatenate([self.z_point, other.z_point]),
                c_plane=np.concatenate([self.c_plane, other.c_plane]),
                r_line=np.concatenate([self.r_line, other.r_line]),
                i_line=np.concatenate([self.i_line, other.i_line]),
                mirror_imag=mirror_imag,
                mirror_real=mirror_real,
            )
        else:
            return self.__class__(
                z_point=self.z_point * other,
                c_plane=self.c_plane * other,
                r_line=self.r_line * other,
                i_line=self.i_line * other,
                mirror_imag=self.mirror_imag,
                mirror_real=self.mirror_real,
            )

    def __rmul__(self, other):
        """
        """
        return self.__class__(
            z_point=other * self.z_point,
            c_plane=other * self.c_plane,
            r_line=other * self.r_line,
            i_line=other * self.i_line,
            mirror_imag=self.mirror_imag,
            mirror_real=self.mirror_real,
        )

    def __str__(self):
        return "2π·" + self.normalized_str(divided_by=2*np.pi)

    def normalized_str(
            self,
            divided_by=1,
            real_format_func=None,
    ):
        strs = ["["]
        for s in self.str_iter(divide_by=divided_by, real_format_func=real_format_func):
            strs.append(s + ', ')
        strs.append("]")
        return ''.join(strs)

    def response_lnG(self, X, h=1, lnG=0):
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
                abs_max = np.max(abs_sq(h)) ** 0.5
                h /= abs_max
                lnG += np.log(abs_max)
            return h, lnG

        h, lnG = VfR(self.c_plane, h, lnG)
        h, lnG = VfR(self.r_line, h, lnG)
        h, lnG = VfR(self.i_line, h, lnG)
        h, lnG = VfR(self.z_point, h, lnG)
        if self.mirror_real:
            h, lnG = VfR(self.c_plane.conjugate(), h, lnG)
            h, lnG = VfR(self.i_line.conjugate(), h, lnG)
        if self.mirror_imag:
            # self.mirror_real not true, self.mirror_imag true
            h, lnG = VfR(-self.c_plane, h, lnG)
            h, lnG = VfR(-self.r_line, h, lnG)
            if self.mirror_real:
                h, lnG = VfR(-self.c_plane.conjugate(), h, lnG)
        return h, lnG


class RootClassifiers:
    def __init__(
        self,
        line_atol=1e-6,
        line_rtol=1e-6,
        match_atol=1e-6,
        match_rtol=1e-6,
        zero_atol=1e-10,
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

        def are_imag(r1):
            if abs(r1.imag) < 0.8:
                return abs(np.real(r1)) < line_atol
            else:
                return abs(np.real(r1) / np.imag(r1)) < line_rtol

        def are_zero(r1):
            return abs(np.imag(r1)) < zero_atol

        self.lax_line_tol = 0
        self.line_atol = line_atol
        self.line_rtol = line_rtol
        self.match_atol = match_atol
        self.match_rtol = match_rtol
        self.zero_atol = zero_atol
        self.are_same = np.vectorize(are_same, otypes=[bool])
        self.are_real = np.vectorize(are_real, otypes=[bool])
        self.are_imag = np.vectorize(are_imag, otypes=[bool])
        self.are_zero = np.vectorize(are_zero, otypes=[bool])

    def NC2MR(self, roots_u):
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
                # roots_c.append((r1 + r2) / 2)
                # TODO, this seems to work better, not clear why..
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

    def NC2MI(self, roots_u):
        imag_select = self.are_imag(roots_u)
        roots_i = roots_u[imag_select].imag
        roots_u = roots_u[~imag_select]

        pos_select = roots_u.real > 0
        roots_c_neg = roots_u[~pos_select]
        roots_c_pos = roots_u[pos_select]
        rPB = nearest_pairs(roots_c_pos, -roots_c_neg.conjugate())

        # TODO, put this logic in the NC2MR

        # fill the list with as many pairs as possible
        r12_list_full = rPB.r12_list
        while rPB.r12_list:
            rPB = nearest_pairs(rPB.l1_remain, rPB.l2_remain)
            r12_list_full.extend(rPB.r12_list)
        rPB.r12_list = r12_list_full

        if self.lax_line_tol > 0:
            roots_u = []
            roots_i2 = []

            def check_ins(u):
                if abs(u.imag) > self.lax_line_tol:
                    if abs(u.real) < self.lax_line_tol:
                        roots_i2.append(u.imag)
                    else:
                        roots_u.append(u)
                else:
                    if abs(u.real / u.imag) < self.lax_line_tol:
                        roots_i2.append(u.imag)
                    else:
                        roots_u.append(u)

            for u in rPB.l1_remain:
                check_ins(u)
            for u in rPB.l2_remain:
                check_ins(-u.conjugate())
            roots_i = np.concatenate([roots_i, roots_i2])
        else:
            roots_u = list(rPB.l1_remain) + [-r.conjugate() for r in rPB.l2_remain]

        roots_c = []
        for r1, r2 in rPB.r12_list:
            if self.are_same(r1, r2):
                # roots_c.append((r1 + r2) / 2)
                # TODO, this seems to work better, not clear why..
                roots_c.append(r1)
            else:
                roots_u.append(r1)
                roots_u.append(-r2.conjugate())
        roots_c = np.array(roots_c)
        roots_u = np.array(roots_u)
        return Bunch(
            c=roots_c,
            i=roots_i,
            u=roots_u,
        )


    def MR2MQ(self, roots_c, roots_r, roots_u):
        imag_select = self.are_imag(roots_c)
        roots_i = roots_c[imag_select].imag
        roots_c = roots_c[~imag_select]

        pos_select = roots_c.real > 0
        roots_c_neg = roots_c[~pos_select]
        roots_c_pos = roots_c[pos_select]
        rPB = nearest_pairs(roots_c_pos, -roots_c_neg.conjugate())
        roots_u2 = list(rPB.l1_remain) + [-r.conjugate() for r in rPB.l2_remain]
        roots_c = []
        for r1, r2 in rPB.r12_list:
            if self.are_same(r1, r2):
                # roots_c.append((r1 + r2) / 2)
                # TODO, this seems to work better, not clear why..
                roots_c.append(r1)
            else:
                roots_u2.append(r1)
                roots_u2.append(-r2.conjugate())
        roots_c = np.array(roots_c)
        roots_u = np.concatenate([roots_u, roots_u2])

        select_zero = self.are_zero(roots_r)
        roots_r = roots_r[~select_zero]
        z = np.count_nonzero(select_zero)
        return Bunch(
            z=z,
            c=roots_c,
            r=roots_r,
            i=roots_i,
            u=roots_u,
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
                    b = self.NC2MR(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    b = self.MR2MQ(
                        roots_c=b.c,
                        roots_r=b.r,
                        roots_u=b.u
                    )
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        c_plane=b.c,
                        r_line=b.r,
                        i_line=b.i,
                        z_point=b.z,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
            else:
                def to_rootset(roots, rtype):
                    b = self.NC2MR(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        c_plane=b.c,
                        r_line=b.r,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
        else:
            if time_symm:
                def to_rootset(roots, rtype):
                    b = self.NC2MI(roots)
                    if b.u:
                        raise RuntimeError(f"Unmatched {rtype}")
                    return tRootSet(
                        c_plane=b.c,
                        i_line=b.i,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )
            else:
                def to_rootset(roots, rtype):
                    return tRootSet(
                        c_plane=roots,
                        hermitian=hermitian,
                        time_symm=time_symm, 
                    )

        return to_rootset

default_root_classifier = RootClassifiers()
