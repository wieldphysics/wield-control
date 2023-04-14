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


def percolate_inplace(A, B, C, D, E, which, ranges, keep=True):

    if keep:
        # make the inverse array through argsorts
        argranges = np.argsort([r[0] for r in ranges])
        argranges = np.argsort(argranges)
        ranges = sorted(ranges)

        def percolate1(mat):
            segments = []
            for r in ranges:
                A1, A2 = r
                Sl1 = slice(A1, A2)
                segments.append(np.copy(mat[Sl1, :]))
            past_r = ranges[0]
            past_idx = past_r[0]
            for r in ranges[1:]:
                Sl1 = slice(past_r[1], r[0])
                next_idx = past_idx + r[0] - past_r[1]
                Sl2 = slice(past_idx, next_idx)
                mat[Sl2, :] = mat[Sl1, :]
                past_idx = next_idx
                past_r = r
            Sl1 = slice(past_r[1], mat.shape[0])
            next_idx = past_idx + mat.shape[0] - past_r[1]
            Sl2 = slice(past_idx, next_idx)
            mat[Sl2, :] = mat[Sl1, :]
            past_idx = next_idx
            for idx_range in argranges:
                seg = segments[idx_range]
                next_idx = past_idx + seg.shape[0]
                Sl2 = slice(past_idx, next_idx)
                mat[Sl2, :] = seg
                past_idx = next_idx

        def percolate2(mat):
            segments = []
            for r in ranges:
                A1, A2 = r
                Sl1 = slice(A1, A2)
                segments.append(np.copy(mat[:, Sl1]))
            past_r = ranges[0]
            past_idx = past_r[0]
            for r in ranges[1:]:
                Sl1 = slice(past_r[1], r[0])
                next_idx = past_idx + r[0] - past_r[1]
                Sl2 = slice(past_idx, next_idx)
                mat[:, Sl2] = mat[:, Sl1]
                past_idx = next_idx
                past_r = r
            Sl1 = slice(past_r[1], mat.shape[1])
            next_idx = past_idx + mat.shape[1] - past_r[1]
            Sl2 = slice(past_idx, next_idx)
            mat[:, Sl2] = mat[:, Sl1]
            past_idx = next_idx
            for idx_range in argranges:
                seg = segments[idx_range]
                next_idx = past_idx + seg.shape[1]
                Sl2 = slice(past_idx, next_idx)
                mat[:, Sl2] = seg
                past_idx = next_idx

    else:
        ranges = sorted(ranges)

        def percolate1(mat):
            past_r = ranges[0]
            past_idx = past_r[0]
            for r in ranges[1:]:
                Sl1 = slice(past_r[1], r[0])
                next_idx = past_idx + r[0] - past_r[1]
                Sl2 = slice(past_idx, next_idx)
                mat[Sl2, :] = mat[Sl1, :]
                past_idx = next_idx
                past_r = r
            Sl1 = slice(past_r[1], mat.shape[0])
            next_idx = past_idx + mat.shape[0] - past_r[1]
            Sl2 = slice(past_idx, next_idx)
            mat[Sl2, :] = mat[Sl1, :]

        def percolate2(mat):
            past_r = ranges[0]
            past_idx = past_r[0]
            for r in ranges[1:]:
                Sl1 = slice(past_r[1], r[0])
                next_idx = past_idx + r[0] - past_r[1]
                Sl2 = slice(past_idx, next_idx)
                mat[:, Sl2] = mat[:, Sl1]
                past_idx = next_idx
                past_r = r
            Sl1 = slice(past_r[1], mat.shape[1])
            next_idx = past_idx + mat.shape[1] - past_r[1]
            Sl2 = slice(past_idx, next_idx)
            mat[:, Sl2] = mat[:, Sl1]

    if which == "inputs":
        percolate2(B)
        percolate2(D)
    elif which == "output":
        percolate1(C)
        percolate1(D)
    elif which == "states":
        percolate2(C)
        percolate2(A)
        percolate2(E)
    elif which == "constr":
        percolate1(B)
        percolate1(A)
        percolate1(E)
    else:
        raise RuntimeError("Unrecognized 'which' argument")
    return
