#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions to print state space matrices
"""
import numpy as np

def nz(M):
    return 1 * (M != 0)


def print_dense_nonzero(ssb):
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
    if ssb.E is not None:
        Estr = np.array2string(nz(ssb.E), max_line_width=np.nan, threshold=30 ** 2)
    else:
        Estr = ''
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
