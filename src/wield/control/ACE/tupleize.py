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
    def __str__(self):
        return detuplize_full(self)

    def __repr__(self):
        return detuplize_full(self)

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


def _tupleize(param):
    if isinstance(param, str):
        return KeyTuple(param.split("."))
    elif isinstance(param, (list, tuple)):
        ps = []
        for p in param:
            tp = _tupleize(p)
            if isinstance(tp, tuple):
                ps.extend(tp)
            else:
                ps.append(tp)
        return KeyTuple((KeyTuple(ps),))
    else:
        return KeyTuple((param,))


def tupleize(param):
    if param is None:
        return KeyTuple()
    if isinstance(param, (list, tuple)):
        # since tupleize re-wraps tuples, undo that on the topmost one
        return _tupleize(param)[0]
    else:
        return _tupleize(param)


def detuplize(ptup):
    no_tuples = True
    for p in ptup:
        if isinstance(p, tuple):
            no_tuples = False
            break
    if no_tuples:
        return ".".join(str(p) for p in ptup)
    else:
        return tuple(detuplize(p) for p in ptup)


def detuplize_full(ptup):
    no_tuples = True
    if ptup is None:
        return "None"
    for p in ptup:
        if isinstance(p, tuple):
            no_tuples = False
            break
    if no_tuples:
        return ".".join(str(p) for p in ptup)
    else:
        return "(" + ",".join(detuplize_full(p) for p in ptup) + ")"


def detuplize_mapping(xx2yy):
    xx2yy2 = dict()
    for xx, yy_set in xx2yy.items():
        xx2yy2[detuplize(xx)] = set(detuplize(yy) for yy in yy_set)
    return xx2yy2


def detuplize_keypairs(xxyymap):
    xxyymap2 = dict()
    for (xx, yy), v in xxyymap.items():
        xxyymap2[detuplize(xx), detuplize(yy)] = v
    return xxyymap2


@functools.total_ordering
class EdgeTuple(namedtuple("EdgeBase", ("r", "c"))):
    def __str__(self):
        return "<{R}&{C}>".format(R=self.r, C=self.c)

    def __repr__(self):
        return "<{R}&{C}>".format(R=self.r, C=self.c)

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
