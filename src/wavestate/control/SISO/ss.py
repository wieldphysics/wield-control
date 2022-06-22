#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions to create a SISO state space system from inputs.
"""
import numbers
import numpy as np

from ..statespace.dense import xfer_algorithms
from ..statespace.dense import zpk_algorithms
from ..statespace.dense import ss_algorithms

from . import siso
from . import zpk


class SISOStateSpace(siso.SISO):
    """
    ZPK class to represent SISO Transfer functions.

    This class internally uses the s-domain in units of radial frequency and gain.
    """
    def __init__(
        self,
        A, B, C, D, E,
        hermitian: bool = True,
        time_symm: bool = False,
        dt=None,
    ):
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        if E is not None:
            E = np.asarray(E)

        if hermitian:
            assert(np.all(A.imag == 0))
            assert(np.all(B.imag == 0))
            assert(np.all(C.imag == 0))
            assert(np.all(D.imag == 0))
            if E is not None:
                assert(np.all(E.imag == 0))

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt

    @property
    def ABCDE(self):
        if self.E is None:
            E = np.eye(self.A.shape[-1])
        else:
            E = self.E
        return self.A, self.B, self.C, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.E

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        yield self.A
        yield self.B
        yield self.C
        yield self.D
        if self.E is not None:
            yield self.E

    _ZPK = None

    @property
    def asZPK(self):
        if self._ZPK is not None:
            self._ZPK
        z, p = zpk_algorithms.ss2zp(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            idx_in=0,
            idx_out=0,
            fmt="scipy",
        )
        self._ZPK = zpk.zpk(
            z, p,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            convention='scipy',
            response=self.response,
        )
        return self._ZPK

    @property
    def asSS(self):
        return self

    def print_nonzero(self):
        """
        """
        return printSSBnz(self)

    def response(self, f=None, w=None, s=None):
        domain = None
        if f is not None:
            domain = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(domain is None)
            domain = 1j * np.asarray(w)
        if s is not None:
            assert(domain is None)
            domain = np.asarray(s)

        return xfer_algorithms.ss2response_siso(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            s=domain,
            idx_in=0,
            idx_out=0,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asSS
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            assert(self.dt == other.dt)
            ABCDE = ss_algorithms.chain([self.ABCDE, other.ABCDE])
            return self.__class__(
                A=ABCDE.A,
                B=ABCDE.B,
                C=ABCDE.C,
                D=ABCDE.D,
                E=ABCDE.E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B * other,
                C=self.C,
                D=self.D * other,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:

            return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=other * self.C,
                D=other * self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented


def ss(
    *args,
    hermitian=True,
    time_symm=False,
    dt=None,
):
    """
    Form a SISO LTI system from statespace matrices.

    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, siso.SISO):
            arg = arg.asSS
        if isinstance(arg, SISOStateSpace):
            return arg
        elif isinstance(arg, (tuple, list)):
            A, B, C, D, E = arg
    elif len(args) == 4:
        A, B, C, D = args
        E = None
    elif len(args) == 5:
        A, B, C, D, E = args
    else:
        raise RuntimeError("Unrecognized argument format")
    return SISOStateSpace(
        A, B, C, D, E,
        hermitian=True,
        time_symm=False,
        dt=None,
    )



def nz(M):
    return 1 * (M != 0)

def printSSBnz(ssb):
    """
    print the nonzero sparsity patter of the statespace.

    NOTE, the code is slightly convoluted as it is based on similar code in the ACE system
    """
    c_str = []
    for key in range(ssb.A.shape[-2]):
        i1 = key
        i2 = i1 + 1
        if i2 - i1 == 0:
            continue
        c_str.append(str(key))
        if i2 - i1 > 1:
            for _ in range(i2 - i1 - 2):
                c_str.append("┃")
            c_str.append("┗")
    c_str = "\n".join(c_str)

    s_str = []
    s_str_list = []
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for key in range(ssb.A.shape[-1]):
        kidx = key % len(alpha)
        i1 = key
        i2 = i1 + 1
        if i2 - i1 == 0:
            continue
        s_str_list.append(alpha[kidx] + ": " + str(key))
        s_str.append(alpha[kidx] + " ")
        if i2 - i1 > 1:
            for _ in range(i2 - i1 - 2):
                s_str.append("━━")
            s_str.append("┓ ")
    s_str = "  " + "".join(s_str)

    Astr = np.array2string(nz(ssb.A), max_line_width=np.nan, threshold=100 ** 2)
    Estr = np.array2string(nz(ssb.E), max_line_width=np.nan, threshold=30 ** 2)
    Bstr = np.array2string(nz(ssb.B), max_line_width=np.nan, threshold=100 ** 2)
    Cstr = np.array2string(nz(ssb.C), max_line_width=np.nan, threshold=100 ** 2)
    Dstr = np.array2string(nz(ssb.D), max_line_width=np.nan, threshold=100 ** 2)
    # print(" | ".join(s_str_list))
    ziplines(
        "\n" + c_str,
        s_str + "\n" + Astr + "\n\n" + Cstr,
        "\n" + Bstr + "\n\n" + Dstr,
        "\n" + Estr,
        delim=" | ",
    )


def ziplines(*args, delim=""):
    import itertools

    widths = []
    for arg in args:
        w = max(len(line) for line in arg.splitlines())
        widths.append(w)
    for al in itertools.zip_longest(*[arg.splitlines() for arg in args], fillvalue=""):
        line = []
        for a, w in zip(al, widths):
            line.append(a + " " * (w - len(a)))
        print(delim.join(line))
