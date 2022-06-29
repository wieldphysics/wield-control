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
from ..statespace import ssprint

from . import siso
from . import zpk


class SISOStateSpace(siso.SISO):
    """
    class to represent SISO Transfer functions using dense state space matrix representations.
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
        return self.A, self.B, self.C, self.D, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.D, self.E

    @property
    def ABCD(self):
        if self.E is None:
            raise RuntimeError("Cannot Drop E")
        else:
            assert(np.all(np.eye(self.E.shape[-1]) == self.E))
            self.E = None
        return self.A, self.B, self.C, self.D

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
        return ssprint.print_dense_nonzero(self)

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


