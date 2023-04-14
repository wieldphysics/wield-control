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
import re
import tabulate


class MIMOTable(object):
    def __init__(
        self,
        table,
        rownames,
        colnames,
        colunits=None,
        rowunits=None,
    ):
        self.table = np.asarray(table)
        self.rownames = list(rownames)
        self.colnames = list(colnames)
        if colunits is None:
            colunits = [None] * len(self.colnames)
        self.colunits = colunits
        if rowunits is None:
            rowunits = [None] * len(self.rownames)
        self.rowunits = rowunits
        return

    def copy(self):
        other = self.__class__.__new__(self.__class__)
        other.table = np.copy(self.table)
        other.rownames = list(self.rownames)
        other.colnames = list(self.colnames)
        other.colunits = list(self.colunits)
        other.rowunits = list(self.rowunits)
        return other

    def sort(self, rownames=None, colnames=None):
        """
        Sorts the table by rows or columns

        The inputs are the rownames or colnames which have been sorted.
        If either is not specified, then the existing order is used
        """
        if colnames is None:
            colnames = self.colnames
        if rownames is None:
            rownames = self.rownames
        rowidx = np.array([self.rownames.index(r) for r in rownames])
        colidx = np.array([self.colnames.index(c) for c in colnames])

        self.rownames = [self.rownames[i] for i in rowidx]
        self.colnames = [self.colnames[i] for i in colidx]

        self.rowunits = [self.rowunits[i] for i in rowidx]
        self.colunits = [self.colunits[i] for i in colidx]

        self.table = self.table[..., rowidx, :][..., :, colidx]
        return

    def push_col(self, colname):
        """
        Pushes a column to the back of the table
        """
        col = self.colnames.index(colname)
        self.colnames[col:-1] = self.colnames[col + 1 :]
        self.colnames[-1] = colname

        colunit = self.colunits[col]
        self.colunits[col:-1] = self.colunits[col + 1 :]
        self.colunits[-1] = colunit

        vec = np.copy(self.table[..., :, col])
        self.table[..., :, col:-1] = self.table[..., :, col + 1 :]
        self.table[..., :, -1] = vec

    def push_row(self, rowname):
        """
        Pushes a row to the back of the table
        """
        row = self.rownames.index(rowname)
        self.rownames[row:-1] = self.rownames[row + 1 :]
        self.rownames[-1] = rowname

        rowunit = self.rowunits[row]
        self.rowunits[row:-1] = self.rowunits[row + 1 :]
        self.rowunits[-1] = rowunit

        vec = np.copy(self.table[..., row, :])
        self.table[..., row:-1, :] = self.table[..., row + 1 :, :]
        self.table[..., -1, :] = vec

    def cut(self, threshold, replace=None, inplace=False):
        """
        Apply a threshold and cutoff to the table. This will hide elements that are under the cutoff

        threshold represents a ratio of each element to the max of its column or the max of its row.
        """
        if not inplace:
            self = self.copy()
        abstable = abs(self.table)
        row = np.max(abstable, axis=-2)
        abstable = abstable / row.reshape(1, -1)
        select_bad_r = abstable < threshold

        col = np.max(abstable, axis=-1)
        abstable = abstable / col.reshape(-1, 1)
        select_bad_c = abstable < threshold

        select_bad = select_bad_r & select_bad_c
        if replace is None:
            self.table = np.array(self.table, dtype=object)
            self.table[..., select_bad] = None
        else:
            self.table[..., select_bad] = replace
        if not inplace:
            return self
        else:
            return

    def loop_close(self, row, col, pushrow=True, pushcol=True):
        """
        Apply an infinite-gain loop from the row sensor to the col actuator.
        This then propagates the cross couplings to give the residual couplings
        of other matrix entrees.
        """
        rowname = row
        colname = col
        row = self.rownames.index(rowname)
        col = self.colnames.index(colname)

        subtable = 0 * self.table

        d = self.table[row, col]
        for idx_row in range(self.table.shape[-2]):
            for idx_col in range(self.table.shape[-1]):
                b = self.table[row, idx_col]
                c = self.table[idx_row, col]
                subtable[..., idx_row, idx_col] = b * c / d

        newtable = self.table - subtable

        newrow = self.rownames[row] = colname + "-CL"
        newcol = self.colnames[col] = rowname + "-SG"

        self.colunits[col], self.rowunits[row] = self.rowunits[row], self.colunits[col]

        # new b'
        for idx_row in range(self.table.shape[-2]):
            if row == idx_row:
                continue
            newtable[idx_row, col] = -self.table[idx_row, col] / d

        # new c'
        for idx_col in range(self.table.shape[-1]):
            if col == idx_col:
                continue
            newtable[row, idx_col] = -self.table[row, idx_col] / d

        # new d'
        newtable[row, col] = -1 / d

        self.table = newtable

        if pushrow:
            self.push_row(newrow)
        if pushcol:
            self.push_col(newcol)

    def tabulate(
        self,
        diag=None,
        units=True,
        tablefmt="simple",
        transform=None,
        drop_character="↓",
    ):
        def transform_default(x):
            if x is None:
                return ""
            elif x == 0:
                return "0"
            elif abs(x) > 1000:
                return "{:> .4e}".format(x)
            elif abs(x) < 1e-3:
                return "{:> .4e}".format(x)
            else:
                v = "{:> .4f}".format(x)
                if v[0] == "-":
                    return " " + v
                return v

        if transform is None:
            if tablefmt == "fancylatex":
                tablefmt = "latex"
            elif tablefmt == "simple":
                transform = transform_default
            else:
                transform = transform_default
        if diag is not None:
            headers = [diag]
        else:
            headers = []

        rows = []

        if units:
            rows.append(
                ["", "[SI]"] + [(u if u is not None else "") for u in self.colunits]
            )
            headers = headers + [""]

        colrenames = []
        lmap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for cidx, cname in enumerate(self.colnames):
            crname = "col({})".format(lmap[cidx])
            colrenames.append(crname)
            headers.append(crname)

        for rlabel, runit, rowvals in zip(self.rownames, self.rowunits, self.table):
            row = [rlabel]
            if runit is None:
                row.append("")
            else:
                row.append(runit)
            row.extend([transform(r) for r in rowvals])
            rows.append(row)

        if tablefmt == "fancylatex":
            tablefmt = "latex"
        stable = tabulate.tabulate(
            rows,
            headers=headers,
            tablefmt=tablefmt,
            numalign="decimal",
            stralign="decimal",
        )

        maxlen = np.max([len(n) for n in self.colnames])
        # print("Maxlen: ", maxlen)
        if colrenames:
            if tablefmt == "simple":
                stable_lines = stable.split("\n")
                header = stable_lines[0]
                dashes = stable_lines[1]
                split_idxs = (
                    np.asarray([m.start() for m in re.finditer(" -", dashes)][1:]) + 1
                )

                sep = 1
                while True:
                    maxlen_pract = np.max(split_idxs[sep:] - split_idxs[:-sep]) - 1
                    if maxlen_pract > maxlen:
                        break
                    sep += 1
                # print("Mlen_pract:", maxlen_pract, " with sep, ", sep)
                headers = [header[: split_idxs[0]]]
                for idx in range(1, sep):
                    headers.append("")

                for idx_name, cname in enumerate(self.colnames):
                    idx_header = idx_name % sep
                    header_start_idx = split_idxs[idx_name]
                    header = headers[idx_header]
                    header = (
                        header
                        + (" " * (header_start_idx - len(header)))
                        + drop_character
                        + cname
                    )
                    headers[idx_header] = header

                # print(split_idxs)
                return "\n".join(headers + stable_lines[1:])
            else:
                newtab = []
                nzip = list(zip(colrenames, self.colnames))[::-1]
                try:
                    while True:
                        newrow = []
                        for i in range(4):
                            crname, cname = nzip.pop()
                            newrow.append("{}| {}".format(crname, cname))
                        newtab.append(newrow)
                except IndexError:
                    if newrow:
                        newtab.append(newrow)

                ntable = tabulate.tabulate(
                    newtab,
                    tablefmt="plain",
                )
                return ntable + "\n" + stable
        else:
            return stable
