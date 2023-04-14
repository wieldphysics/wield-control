#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import numpy as np
import enum

from wield.utilities.np import matrix_stack, vector_stack, broadcast_shapes

ZERO = np.asarray(0)
IDENT = np.asarray(1)


def as_linval(val):
    """
    Convert the object to a linval. Doesn't convert anything to a diagonal type. Mostly discriminates between
    Scalars and matrices
    """
    if isinstance(val, LinearValue):
        return val
    elif isinstance(val, np.ndarray):
        if val.shape == ():
            if val == 0:
                return LinearValue(LVtype=LinearValueTypes.ZERO, value=ZERO)
            elif val == 1:
                return LinearValue(LVtype=LinearValueTypes.IDENT, value=IDENT)
            else:
                return LinearValue(LVtype=LinearValueTypes.SCALAR, value=val)
        else:
            return LinearValue(LVtype=LinearValueTypes.MATRIX, value=val)
    elif isinstance(val, list):
        return matrix(val)
    else:
        if val == 0:
            return LinearValue(LVtype=LinearValueTypes.ZERO, value=ZERO)
        elif val == 1:
            return LinearValue(LVtype=LinearValueTypes.IDENT, value=IDENT)
        else:
            return LinearValue(LVtype=LinearValueTypes.SCALAR, value=np.asarray(val))


def scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return LinearValue(LVtype=LinearValueTypes.SCALAR, value=value)

    if value == 0:
        return LinearValue(LVtype=LinearValueTypes.ZERO, value=ZERO)
    elif value == 1:
        return LinearValue(LVtype=LinearValueTypes.IDENT, value=IDENT)
    else:
        return LinearValue(LVtype=LinearValueTypes.SCALAR, value=np.asarray(value))


def diagonal(value):
    if isinstance(value, list):
        value = vector_stack(value)
    return LinearValue(LVtype=LinearValueTypes.DIAGONAL, value=np.asarray(value))


def matrix(value):
    if isinstance(value, list):
        value = matrix_stack(value)
    return LinearValue(LVtype=LinearValueTypes.MATRIX, value=np.asarray(value))


class LinearValue(object):
    """
    This object holds a scalar, diagonal, or dense matrix.

    Scalars can be either unwrapped by numpy, or they have dimension () or (...)
    Matrices are stored by numpy arrays and have dimension (...,M,N)
    diagonals are stored by numpy arrays and have dimension (...,M) or  (...,N)
    """

    def __init__(self, LVtype, value):
        """
        Takes the LVtype (if it's a scalar, diagonal, or matrix). Also takes
        the actual matrix (built however it should be)
        """
        assert(LVtype in LinearValueTypes)
        self.LVtype = LVtype
        self.value = value
        assert(isinstance(value, np.ndarray))

        if __debug__:
            if LVtype == LinearValueTypes.DIAGONAL:
                assert(len(value.shape) >= 1)
            elif LVtype == LinearValueTypes.MATRIX:
                assert(len(value.shape) >= 2)

    def __str__(self):
        if self.LVtype == LinearValueTypes.SCALAR:
            return str(self.value)
        elif self.LVtype == LinearValueTypes.DIAGONAL:
            return str(self.value) + ":d{}".format(self.value.shape[-1])
        elif self.LVtype == LinearValueTypes.MATRIX:
            return str(self.value) + ":({},{})".format(self.value.shape[-2], self.value.shape[-1])
        elif self.LVtype == LinearValueTypes.IDENT:
            return str(self.value) + ":"
        elif self.LVtype == LinearValueTypes.ZERO:
            return str(self.value) + ":"
        else:
            raise NotImplementedError()
        return

    def __repr__(self):
        return repr(self.value)

    def __getitem__(self, key):
        """ """
        return self.value[key]

    def __setitem__(self, key, val):
        """ """
        self.value[key] = val
        return

    def __add__(self, other):
        other = as_linval(other)
        return self.matadd(other)

    def __radd__(self, other):
        other = as_linval(other)
        return other.matadd(self)

    def __sub__(self, other):
        other = as_linval(other)
        return self.matsub(other)

    def __rsub__(self, other):
        other = as_linval(other)
        return other.matsub(self)

    def __eq__(self, other):
        if self.LVtype != other.LVtype:
            return False
        return self.value == other.value

    def __neg__(self):
        # needs the asarray since apparently numpy will
        # unwrap some values on the unary negative operator
        if self.LVtype == LinearValueTypes.IDENT:
            return LinearValue(
                LVtype=LinearValueTypes.SCALAR,
                value=np.asarray(-self.value),
            )
        else:
            return LinearValue(
                LVtype=self.LVtype,
                value=np.asarray(-self.value),
            )

    def matadd(self, other):
        assert isinstance(other, LinearValue)

        if other.LVtype == LinearValueTypes.ZERO:
            return LinearValue(
                LVtype=self.LVtype,
                value=self.value,
            )

        if self.LVtype == LinearValueTypes.MATRIX:
            LVtype = LinearValueTypes.MATRIX
            if other.LVtype == LinearValueTypes.MATRIX:
                C = self.value + other.value
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                subshape = broadcast_shapes([self.value.shape[:-2], other.value.shape[:-1]])
                C = np.copy(np.broadcast_to(self.value, subshape + self.value.shape[-2:]))
                # could be faster, but doesn't assume commutativity
                C.reshape(*subshape, -1)[..., :: C.shape[0] + 1] = (
                    other.value + C.reshape(*subshape, -1)[..., :: C.shape[0] + 1]
                )
            else:
                subshape = broadcast_shapes([self.value.shape[:-2], other.value.shape])
                C = np.copy(np.broadcast_to(self.value, subshape + self.value.shape[-2:]))
                # could be faster, but doesn't assume commutativity
                C.reshape(*subshape, -1)[..., :: C.shape[0] + 1] = (
                    other.value.reshape(*other.value.shape, 1) + C.reshape(*subshape, -1)[..., :: C.shape[0] + 1]
                )

        elif self.LVtype == LinearValueTypes.DIAGONAL:
            if other.LVtype == LinearValueTypes.MATRIX:
                # other.value is a matrix here
                subshape = broadcast_shapes([other.value.shape[:-2], self.value.shape[:-1]])
                C = np.copy(np.broadcast_to(other.value, subshape + other.value.shape[-2:]))
                # could be faster, but doesn't assume commutativity
                C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1] = (
                    self.value + C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1]
                )
                LVtype = LinearValueTypes.MATRIX
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value + other.value
                LVtype = LinearValueTypes.DIAGONAL
            else:
                C = self.value + other.value.reshape(*other.value.shape, 1)
                LVtype = LinearValueTypes.DIAGONAL

        elif self.LVtype == LinearValueTypes.SCALAR:
            if other.LVtype == LinearValueTypes.MATRIX:
                subshape = broadcast_shapes([other.value.shape[:-2], self.value.shape])
                C = np.copy(np.broadcast_to(other.value, subshape + other.value.shape[-2:]))
                C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1] = (
                    self.value.reshape(*self.value.shape, 1) + C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1]
                )
                LVtype = LinearValueTypes.MATRIX
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value.reshape(*self.value.shape, 1) + other.value
                LVtype = other.LVtype
            else:
                C = self.value + other.value
                LVtype = other.LVtype

        elif self.LVtype == LinearValueTypes.IDENT:
            if other.LVtype == LinearValueTypes.MATRIX:
                subshape = other.value.shape[:-2]
                C = np.copy(np.broadcast_to(other.value, subshape + other.value.shape[-2:]))
                C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1] = (
                    1 + C.reshape(*subshape, -1)[..., :: C.shape[-2] + 1]
                )
                LVtype = LinearValueTypes.MATRIX
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value.reshape(*self.value.shape, 1) + other.value
                LVtype = other.LVtype
            else:
                C = self.value + other.value
                LVtype = other.LVtype

        elif self.LVtype == LinearValueTypes.ZERO:
            C = other.value
            LVtype = other.LVtype

        else:
            raise NotImplementedError()

        # fixes a reversion from array types for scalars
        if LVtype == LinearValueTypes.SCALAR:
            C = np.asarray(C)

        return LinearValue(
            LVtype=LVtype,
            value=C,
        )

    def matsub(self, other):
        assert isinstance(other, LinearValue)

        # For now, do the slow thing and negate first
        # TODO copy matadd logic and make it subtract
        return self.matadd(-other)

    def __matmul__(self, other):
        other = as_linval(other)
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = as_linval(other)
        return other.matmul(self)

    def matmul(self, other):
        assert isinstance(other, LinearValue)

        if other.LVtype == LinearValueTypes.ZERO:
            return LinearValue(
                LVtype=other.LVtype,
                value=other.value,
            )
        elif other.LVtype == LinearValueTypes.IDENT:
            return LinearValue(
                LVtype=self.LVtype,
                value=self.value,
            )

        if self.LVtype == LinearValueTypes.SCALAR:
            if other.LVtype == LinearValueTypes.MATRIX:
                C = self.value.reshape(*self.value.shape, 1, 1) * other.value
                LVtype = other.LVtype
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value.reshape(*self.value.shape, 1) * other.value
                LVtype = other.LVtype
            elif other.LVtype == LinearValueTypes.SCALAR:
                C = self.value * other.value
                LVtype = other.LVtype

        elif self.LVtype == LinearValueTypes.DIAGONAL:
            if other.LVtype == LinearValueTypes.MATRIX:
                C = self.value.reshape(*self.value.shape, 1) * other.value
                LVtype = LinearValueTypes.MATRIX
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value * other.value
                LVtype = LinearValueTypes.DIAGONAL
            elif other.LVtype == LinearValueTypes.SCALAR:
                C = self.value * other.value
                LVtype = LinearValueTypes.DIAGONAL

        elif self.LVtype == LinearValueTypes.MATRIX:
            LVtype = LinearValueTypes.MATRIX
            if other.LVtype == LinearValueTypes.MATRIX:
                C = self.value @ other.value
            elif other.LVtype == LinearValueTypes.DIAGONAL:
                C = self.value * other.value.reshape(*other.value.shape[:-1], 1, other.value.shape[-1])
            elif other.LVtype == LinearValueTypes.SCALAR:
                C = self.value * other.value

        elif self.LVtype == LinearValueTypes.IDENT:
            C = other.value
            LVtype = other.LVtype

        elif self.LVtype == LinearValueTypes.ZERO:
            C = ZERO
            LVtype = LinearValueTypes.ZERO

        else:
            raise NotImplementedError()

        # fixes a reversion from array types for scalars
        if LVtype == LinearValueTypes.SCALAR:
            C = np.asarray(C)

        return LinearValue(
            LVtype=LVtype,
            value=C,
        )

    def matmuladd(self, othermul, otheradd):
        # assert(isinstance(othermul, LinearValue))
        # assert(isinstance(otheradd, LinearValue))
        # TODO, actually implement this with the proper fast operation
        # in comments above
        return self.matmul(othermul).matadd(otheradd)

    def __pow__(self, other):
        if self.LVtype == LinearValueTypes.SCALAR:
            return LinearValue(
                LVtype=self.LVtype,
                value=np.asarray(self.value**other),
            )
        elif self.LVtype == LinearValueTypes.DIAGONAL:
            return LinearValue(
                LVtype=self.LVtype,
                value=self.value**other,
            )
        elif self.LVtype == LinearValueTypes.MATRIX:
            if other == 1:
                return LinearValue(
                    LVtype=self.LVtype,
                    value=self.value,
                )
            elif other == -1:
                return LinearValue(
                    LVtype=self.LVtype,
                    value=np.linalg.inv(self.value),
                )
            else:
                return LinearValue(
                    LVtype=self.LVtype,
                    value=np.linalg.matrix_power(self.value, other),
                )
        elif self.LVtype == LinearValueTypes.ZERO:
            return LinearValue(
                LVtype=self.LVtype,
                value=ZERO,
            )
        elif self.LVtype == LinearValueTypes.IDENT:
            return LinearValue(
                LVtype=self.LVtype,
                value=IDENT,
            )
        else:
            raise NotImplementedError()

    def matinv(self):
        if self.LVtype == LinearValueTypes.SCALAR:
            return LinearValue(
                LVtype=LinearValueTypes.SCALAR,
                value=1 / self.value,
            )
        elif self.LVtype == LinearValueTypes.DIAGONAL:
            return LinearValue(
                LVtype=LinearValueTypes.DIAGONAL,
                value=1 / self.value,
            )
        elif self.LVtype == LinearValueTypes.MATRIX:
            return LinearValue(
                LVtype=LinearValueTypes.MATRIX,
                value=np.linalg.inv(self.value),
            )
        elif self.LVtype == LinearValueTypes.IDENT:
            return LinearValue(
                LVtype=LinearValueTypes.IDENT,
                value=IDENT,
            )
        elif self.LVtype == LinearValueTypes.ZERO:
            return LinearValue(
                LVtype=LinearValueTypes.ZERO,
                value=ZERO,
            )
        else:
            raise NotImplementedError()

    def abs_sq(self):
        return self.value.real**2 + self.value.imag**2

    def __abs__(self):
        return self.abs_sq()**0.5


class LinearValueTypes(enum.Enum):
    """
    """
    ZERO = 0
    IDENT = 1
    SCALAR = 2
    MATRIX = 3
    DIAGONAL = 4
