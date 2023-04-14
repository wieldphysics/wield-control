#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import re

from ..import string_tuple_keys as stk
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import sympy as sp
except ImportError:
    sp = None


RE_VEC = re.compile(r"\((.*)<(.*)\)")


def yamlstr_convert(a):
    # TODO, make this a little less fragile for generalized edges
    # also check th
    if isinstance(a, stk.KeyTuple):
        a = tuple(a)
    elif isinstance(a, stk.EdgeTuple):
        a = "({}<{})".format(a.r, a.c)
    elif isinstance(a, str):
        pass
    else:
        raise a
    return a


def yamlstr_convert_rev(s):
    m = RE_VEC.match(s)
    if m:
        a = stk.key_edge(m.group(1), m.group(2))
    else:
        a = stk.key_map(s)
    return a


def normalize_list2tuple(v):
    if isinstance(v, list):
        v = tuple(normalize_list2tuple(i) for i in v)
    return v


def to_label(val):
    if not val:
        return ""
    if sp is not None:
        return '$' + sp.latex(sp.var(str(val))) + '$'
    return val
