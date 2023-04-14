#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""

"""
import functools
from collections import namedtuple


@functools.total_ordering
class KeyTuple(tuple):
    def __lt__(self, other):
        if isinstance(other, str):
            return False
        else:
            return super().__lt__(other)

    def __gt__(self, other):
        if isinstance(other, str):
            return True
        else:
            return super().__gt__(other)

    def __add__(self, other):
        return self.__class__(super().__add__(other))

    def __radd__(self, other):
        return self.__class__(super().__add__(other))


@functools.total_ordering
class EdgeTuple(namedtuple("EdgeBase", ("r", "c"))):
    def __str__(self):
        return "{{{R}<{C}}}".format(R=self.r, C=self.c)

    def __repr__(self):
        return "{{{R}<{C}}}".format(R=self.r, C=self.c)

    def __lt__(self, other):
        if isinstance(other, str):
            return False
        else:
            return super().__lt__(other)

    def __gt__(self, other):
        if isinstance(other, str):
            return True
        else:
            return super().__gt__(other)


def key_map(A):
    if A is None:
        raise RuntimeError("BUG")
    if isinstance(A, str):
        return A
    elif isinstance(A, (tuple, list)):
        return KeyTuple(A)
    else:
        return KeyTuple((A,))

def key_join(A, B):
    A = key_map(A)
    B = key_map(B)

    if isinstance(A, str):
        if isinstance(B, str):
            # join as strings
            return A + '.' + B
        else:
            A = KeyTuple((A,))
        # join as tuples
        return A + B
    else:
        if isinstance(B, str):
            B = KeyTuple((B,))
        # join as tuples
        return A + B

def key_edge(A, B):
    A = key_map(A)
    B = key_map(B)
    return EdgeTuple(A, B)
