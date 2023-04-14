#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numbers
import numpy as np
# import warnings

from ..algorithms.statespace.dense import xfer_algorithms
from ..algorithms.statespace.dense import zpk_algorithms
from ..algorithms.statespace.dense import ss_algorithms
from ..statespace import ssprint


class RawStateSpace(object):
    """
    State space class to represent MIMO Transfer functions using dense matrix representations

    This class uses raw matrix representations and should not generally be used by users.

    It is used internally by the SISO.SISOStateSpace and MIMO.MIMOStateSpace classes
    """
    def __init__(
        self,
        A, B, C, D, E, *,
        hermitian: bool = True,
        time_symm: bool = False,
        flags={},
        dt=None,
        warn=True,
    ):
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        assert(A.shape[-1] == A.shape[-2])
        if E is not None:
            E = np.asarray(E)

        if hermitian:
            assert(np.all(A.imag == 0))
            assert(np.all(B.imag == 0))
            assert(np.all(C.imag == 0))
            assert(np.all(D.imag == 0))
            if E is not None:

                assert(np.all(E.imag == 0))

        self.flags = flags

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt
        return

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
    def Ninputs(self):
        return self.B.shape[-1]

    @property
    def Noutputs(self):
        return self.C.shape[-2]

    @property
    def Nstates(self):
        return self.A.shape[-1]

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
        Represent self like a typical scipy ABCD tuple. This throws away symmetry information
        """
        yield self.A
        yield self.B
        yield self.C
        yield self.D
        if self.E is not None:
            yield self.E

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self)

    _p_vals = None

    @property
    def _p(self):
        """
        Create a raw z, p tuple from the direct calculation
        """
        # TODO, not sure this should be included
        if self._p_vals is None:
            p = zpk_algorithms.ss2p(
                A=self.A,
                B=self.B,
                C=self.C,
                D=self.D,
                E=self.E,
                fmt="scipy",
                allow_MIMO=True,
            )
            self._p_vals = p
        return self._p_vals

    def __getitem__(self, key):
        """
        key must be a tuple of a list of row and column elements.

        It can also be tuple of slices
        """
        row, col = key

        ret = self.__class__(
            A=self.A,
            B=self.B[..., :, col],
            C=self.C[..., row, :],
            # double index fixes annoying way that multiple list indices are grouped by numpy
            D=self.D[..., row, :][..., :, col],
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        return ret

    def fresponse_raw(self, *, f=None, w=None, s=None, z=None):
        # TODO fix this import
        from ..SISO import util
        domain = util.build_sorz(
            f=f,
            w=w,
            s=s,
            z=z,
            dt=self.dt,
        )
        return xfer_algorithms.ss2response_mimo(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            sorz=domain,
        )

    def feedbackD(self, D):
        """
        Feedback linkage for a single statespace

        connections_rowcol is a list of row, col pairs
        gain is the connection gain to apply
        """

        fbD = D
        clD = np.linalg.solve(np.eye(self.D.shape[-1]) - fbD @ self.D, fbD)

        if self.dt is not None:
            raise NotImplementedError("feedback not yet implemented in discrete time")

        A = self.A + self.B @ clD @ self.C
        B = self.B + self.B @ clD @ self.D
        C = self.C + self.D @ clD @ self.C
        D = self.D + self.D @ clD @ self.D

        return self.__class__(
            A,
            B,
            C,
            D,
            self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def __matmul__(self, other):
        """
        """
        if isinstance(other, RawStateSpace):
            # currently need to do some checking about the inputs
            # and the outputs
            #return NotImplemented

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
        else:
            return NotImplemented

    def is_square(self):
        return self.D.shape[-1] == self.D.shape[-2]

    def square_size(self):
        assert(self.is_square())
        return self.D.shape[-1]

    def inv(self):
        assert(self.is_square)
        ABCDE = ss_algorithms.inverse_DSS(*self.ABCDE)
        return self.__class__(
            A=ABCDE.A,
            B=ABCDE.B,
            C=ABCDE.C,
            D=ABCDE.D,
            E=ABCDE.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
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

    def __truediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=self.C / other,
                D=self.D / other,
                E=self.E
                ,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __pow__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            if other == -1:
                return self.inv()
            elif other == 1:
                return self
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __add__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            other = _number2D_like(self, other)
            # convert to statespace form
            knownSS = True

        if knownSS or isinstance(other, RawStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)
            assert(self.dt == other.dt)

            return self.__class__(
                A=A,
                B=np.block([
                    [self.B],
                    [other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=self.D + other.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        return NotImplemented

    def __radd__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            other = _number2D_like(self, other)
            # convert to statespace form
            return other + self

        return NotImplemented

    def __sub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            other = _number2D_like(self, other)
            # convert to statespace form
            knownSS = True

        if knownSS or isinstance(other, RawStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)

            return self.__class__(
                A=A,
                B=np.block([
                    [self.B],
                    [-other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=self.D - other.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
            )
        return NotImplemented

    def __rsub__(self, other):
        """
        """
        knownSS = False
        if isinstance(other, numbers.Number):
            # convert to statespace form
            other = _number2D_like(self, other)
            knownSS = True

        if knownSS or isinstance(other, RawStateSpace):
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm

            A, E = joinAE(self, other)

            return self.__class__(
                A=A,
                B=np.block([
                    [-self.B],
                    [other.B]
                ]),
                C=np.block([[self.C, other.C]]),
                D=other.D - self.D,
                E=E,
                hermitian=hermitian,
                time_symm=time_symm,
            )
        return NotImplemented


class RawStateSpaceUser(object):
    def __init__(self, *, ss):
        self.ss = ss

    @property
    def A(self):
        return self.ss.A

    @property
    def B(self):
        return self.ss.B

    @property
    def C(self):
        return self.ss.C

    @property
    def D(self):
        return self.ss.D

    @property
    def E(self):
        return self.ss.E

    @property
    def ABCDE(self):
        return self.ss.ABCDE

    @property
    def ABCDe(self):
        return self.ss.ABCDe

    @property
    def dt(self):
        return self.ss.dt

    @property
    def hermitian(self):
        return self.ss.hermitian

    @property
    def time_symm(self):
        return self.ss.time_symm

    @property
    def structure_flags(self):
        return self.ss.flags

    @property
    def ABCD(self):
        return self.ss.ABCD

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        return iter(self.ss)

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self.ss)


def joinAE(s, o):
    """
    Perform a join on the A and E matrices for two statespaces.
    This is used for the binary add and sub operations
    """
    blU = np.zeros((s.A.shape[-2], o.A.shape[-1]))
    blL = np.zeros((o.A.shape[-2], s.A.shape[-1]))

    if s.E is None and o.E is None:
        E = None
    else:
        if s.E is None:
            sE = np.eye(s.A.shape[-2])
            oE = o.E
        elif o.E is None:
            sE = s.E
            oE = np.eye(o.A.shape[-2])
        else:
            sE = s.E
            oE = o.E
        E = np.block([
            [sE,  blU],
            [blL, oE]
        ]),

    A = np.block([
        [s.A,  blU],
        [blL, o.A]
    ])
    return A, E


def _number2D_like(self, other):
    other = np.asarray(other)
    assert(self.is_square())
    size = self.square_size()
    other_D = np.eye(size) * other
    other = self.__class__(
        A=np.array([[]]).reshape(0, 0),
        B=np.array([[]]).reshape(0, size),
        C=np.array([[]]).reshape(size, 0),
        D=other_D,
        hermitian=(other.imag == 0),
        time_symm=True,
        dt=self.dt
    )
    return other
