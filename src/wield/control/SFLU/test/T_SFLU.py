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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy
from wield.utilities.mpl import mplfigB
import wield.control
import re

from wield.control.SFLU import SFLU

from wield.pytest import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


def T_SFLU_pre1(dprint, tpath_join, fpath_join):
    """
    Setup 2-node problem, nodes a and b with edges Eaa, Eab, Eba, Ebb.
    """

    edges = {
        ("a", "a_in"): 1,
        ("a_out", "a"): 1,
        ("a", "a"): "Eaa",
        ("b", "a"): "Eba",
        ("a", "b"): "Eab",
        ("b", "b"): "Ebb",
    }
    sflu = SFLU.SFLU(edges)

    print("-----------------------")
    dprint(sflu.edges)
    sflu.reduce("a")
    print("-----------------------")
    dprint(sflu.edges)
    dprint(sflu.row2)
    dprint(sflu.col2)
    sflu.reduce("b")
    print("-----------------------")
    dprint(sflu.edges)
    dprint(sflu.row2)
    dprint(sflu.col2)
    print("-----------------------oplist")
    dprint(sflu.oplistE)
    dprint(sflu.subinverse("a_out", "a_in"))

    return


def T_sympy(pprint):
    x = sympy.Symbol("x", commutative=False)
    y = sympy.Symbol("y", commutative=False)

    def symbols():
        i = 0
        while True:
            yield sympy.Symbol("S_{}".format(i), commutative=False)
            i += 1

    E = x * y ** 2 - (y ** -2 * x) ** -2
    dprint(E)
    subexprs, cse_expr = sympy.cse(E, symbols=symbols())

    def transform(ex):
        return re.sub(r"([^*])\*([^*])", r"\1 @ \2", str(ex), count=0)

    for expr in subexprs:
        print(str(expr[0]), "=", transform(expr[1]))
    print(transform(cse_expr[0]))
