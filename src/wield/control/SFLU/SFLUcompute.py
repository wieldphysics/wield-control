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
import yaml
import re

from collections import defaultdict

from wield.utilities.np import matrix_stack

from ..import string_tuple_keys as stk
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import sympy as sp
except ImportError:
    sp = None

from . import SFLU
from .conversions import (
    yamlstr_convert,
    yamlstr_convert_rev,
    normalize_list2tuple,
)


class SFLUCompute:
    def __init__(
        self,
        oplistE,
        edges,
        row2col,
        col2row,
        typemap_to=None,
        typemap_fr=None,
        eye=1,
    ):
        self.oplistE = oplistE
        self.edges = edges
        self.row2col = row2col
        self.col2row = col2row

        edges_rev = defaultdict(set)

        for edge, v in self.edges:
            edges_rev[v].add(edge)

        self.edges_rev = dict(edges_rev)

        if typemap_to is None:
            typemap_to = self.typemap_to_default
        if typemap_fr is None:
            typemap_fr = self.typemap_fr_default
        self.typemap_to = typemap_to
        self.typemap_fr = typemap_fr
        self.eye = self.typemap_to(eye)
        return

    def CLG_inv(self, E):
        # return np.linalg.inv(self.eye - E)
        return np.linalg.inv(np.eye(E.shape[-1]) - E)

    def typemap_to_default(self, v):
        """
        Default conversion for edge and node values.
        This conversion promotes scalars and arrays to become
        1x1 matrices so that the matmul operation may be applied
        """
        if isinstance(v, list):
            v = matrix_stack(v)
        else:
            v = np.asarray(v)
            if len(v.shape) == 0:
                v = v.reshape(1, 1)
            elif len(v.shape) == 1:
                v = v.reshape(v.shape[0], 1, 1)
        return v

    def typemap_fr_default(self, v):
        if v.shape[-2:] == (1, 1):
            return v[..., 0, 0]
        return v

    def edge_map(self, edge_map, default=False):
        """
        Map the edges into an edge space for computation.
        """
        def edge_compute(ev):
            if isinstance(ev, str):
                # if the value is a string, then we pull it out of the edge map

                # first check if we should
                # perform some mapping operations on the string to evaluate
                # simple mathematical operations like unary negation
                if ev[0] == '-':
                    ev = ev[1:]
                    if default is not False:
                        return self.typemap_to(-edge_map.setdefault(ev, default))
                    else:
                        return self.typemap_to(-edge_map[ev])
                else:
                    # no operation, so directly compute the mapping
                    if default is not False:
                        return self.typemap_to(edge_map.setdefault(ev, default))
                    else:
                        return self.typemap_to(edge_map[ev])
            else:
                # then it must be an edge computation
                op = ev[0]
                if op == '*':
                    prod = edge_compute(ev[1])
                    for v in ev[2:]:
                        prod = prod @ edge_compute(v)
                    return prod
                elif op == '-':
                    if len(ev) == 2:
                        return -edge_compute(ev[1])
                    elif len(ev) == 3:
                        return edge_compute(ev[1])-edge_compute(ev[2])
                    else:
                        assert(False)
                else:
                    raise NotImplementedError("Unrecognized edge computation opcode")
            return

        Espace = {}
        for ek, ev in self.edges.items():
            assert(isinstance(ek, stk.EdgeTuple))
            Espace[ek] = edge_compute(ev)

        return Espace

    def compute(self, edge_map):
        Espace = self.edge_map(edge_map)

        for op in self.oplistE:
            # print(op)
            if op.op == "E_CLG":
                (arg,) = op.args
                E = Espace[arg]
                # assert E.shape[-1] == E.shape[-2]
                # I = np.eye(E.shape[-1])
                # I = I.reshape((1,) * (len(E.shape) - 2) + (E.shape[-2:]))
                E2 = self.CLG_inv(E)

                Espace[op.targ] = E2
                # print("CLG: ", op.targ, E2)

            elif op.op == "E_CLGd":
                # (arg,) = op.args
                # size = self.nodesizes.get(arg, self.defaultsize)
                # I = np.eye(size)
                for ek, ev in Espace.items():
                    if ek.r == op.args[0]:
                        rdim = ev.shape[-2]  # row dimension of incoming edge
                        break
                Espace[op.targ] = np.eye(rdim)
                # Espace[op.targ] = self.eye

            elif op.op == "E_mul2":
                arg1, arg2 = op.args
                E1 = Espace[arg1]
                E2 = Espace[arg2]
                Espace[op.targ] = E1 @ E2

            elif op.op == "E_mul3":
                arg1, arg2, arg3 = op.args
                E1 = Espace[arg1]
                E2 = Espace[arg2]
                E3 = Espace[arg3]
                Espace[op.targ] = E1 @ E2 @ E3

            elif op.op == "E_mul3add":
                arg1, arg2, arg3, argA = op.args
                E1 = Espace[arg1]
                E2 = Espace[arg2]
                E3 = Espace[arg3]
                EA = Espace[argA]

                Espace[op.targ] = (E1 @ E2 @ E3 + EA)
                # print("MUL3ADD: ", op.targ, Espace[op.targ], E1, E2, E3, EA)

            elif op.op == "E_assign":
                (arg,) = op.args
                Espace[op.targ] = Espace[arg]

            elif op.op == "E_del":
                del Espace[op.targ]

            else:
                raise RuntimeError("Unrecognized Op {}".format(op))

        self.Espace = Espace
        return Espace

    def subinverse_by(self, oplistN):
        Nspace = dict()

        for op in oplistN:
            if op.op == "N_edge":
                # load an edge into a node
                (Earg,) = op.args
                Ntarg = Nspace.get(op.targ, None)
                E = self.Espace[Earg]

                if Ntarg is None:
                    Nspace[op.targ] = E.copy()
                elif Ntarg.shape == E.shape:
                    Nspace[op.targ] += E
                else:
                    Nspace[op.targ] = Ntarg + E

            # N_sum does not know if the node has been loaded already
            elif op.op == "N_sum":
                argE, argN = op.args
                Ntarg = Nspace.get(op.targ, None)
                prod = self.Espace[argE] @ Nspace[argN]
                if Ntarg is None:
                    Nspace[op.targ] = prod
                elif Ntarg.shape == prod.shape:
                    Nspace[op.targ] += self.Espace[argE] @ Nspace[argN]
                else:
                    Nspace[op.targ] = Ntarg + self.Espace[argE] @ Nspace[argN]

            elif op.op == "N_ret":
                return Nspace[op.targ]

            else:
                raise RuntimeError("Unrecognized Op {}".format(op))

    def subgraph(self, rows, cols):
        """
        Generate the row2col and col2row for a subgraph
        """
        # here, create a col2row dict with only the subset of nodes needed
        # to reach the requested output Rset
        col2row = defaultdict(set)
        row_stack = list(rows)
        while row_stack:
            rN = row_stack.pop()
            cS = self.row2col[rN]
            for cN in cS:
                rS = col2row[cN]
                if not rS:
                    if cN in self.row2col:
                        row_stack.append(cN)
                rS.add(rN)

        row2col = defaultdict(set)
        col_stack = list(cols)
        while col_stack:
            cN = col_stack.pop()
            rS = col2row[cN]
            for rN in rS:
                cS = row2col[rN]
                if not cS:
                    if rN in col2row:
                        col_stack.append(rN)
                cS.add(cN)
        return row2col, col2row

    def inverse_col_single(self, Rset, C, derivatives=False):
        """
        Find the inverse of a single column into many rows given by Rset

        derivatives: if True, include all derivative testpoints in Rset
        """
        return self.inverse_col(Rset, {C: None}, derivatives=derivatives)

    def inverse_col(self, Rset, Cmap, derivatives=False):
        """
        This computes the matrix element for an inverse from C to R.

        The columns are a dictionary mapping column names to values. The values can be vectors, matrices or None.
        A value of None implicitly chooses the identity matrix.

        TODO, the algorithm could/should use some work. Should make a copy of col2row
        then deplete it

        derivatives: if True, include all derivative testpoints in Rset
        """
        Rset = set(stk.key_map(R) for R in Rset)
        Cmap = {stk.key_map(C): v for C, v in Cmap.items()}

        row2col, col2row = self.subgraph(Rset, Cmap.keys())

        Nspace = dict(Cmap)
        Nnum = dict()
        col_stack = list(Cmap.keys())

        while col_stack:
            cN = col_stack.pop()
            v = Nspace[cN]
            # free space now that v is captured
            if cN not in Rset:
                del Nspace[cN]

            rS = col2row[cN]
            for rN in rS:
                E = self.Espace[rN, cN]
                prev = Nspace.get(rN, None)
                if v is None:
                    addin = E
                else:
                    addin = E @ v
                if prev is None:
                    Nspace[rN] = addin
                    Nnum[rN] = 1
                else:
                    # TODO, could make this in-place
                    Nspace[rN] = prev + addin
                    Nnum[rN] += 1
                # this condition means that the row node has been fully filled by
                # all col nodes
                if Nnum[rN] == len(row2col[rN]):
                    col_stack.append(rN)

        assert(set(Nspace.keys()) == Rset)

        # ops.append(OpComp("N_edge", node, (stk.key_edge(node, cN),)))
        # ops.append(
        #     OpComp("N_sum", node, (stk.key_edge(node, cN), cN))
        # )
        # ops.append(OpComp("N_ret", R, ()))
        return Nspace

    def inverse_row_single(self, R, Cset, derivatives=False):
        """
        Find the inverse of a single row from many columns given by Cset

        derivatives: if True, include all derivative excitations in Cset
        """
        return self.inverse_row({R: None}, Cset, derivatives=derivatives)

    def inverse_row(self, Rmap, Cset, derivatives=False):
        """
        This computes the matrix element for an inverse from C to R

        TODO, the algorithm could/should use some work. Should make a copy of col2row
        then deplete it

        derivatives: if True, include all derivative excitations in Cset
        """
        Cset = set(stk.key_map(C) for C in Cset)
        Rmap = {stk.key_map(R): v for R, v in Rmap.items()}

        row2col, col2row = self.subgraph(Rmap.keys(), Cset)

        Nspace = dict(Rmap)
        Nnum = dict()
        row_stack = list(Rmap.keys())

        while row_stack:
            rN = row_stack.pop()
            v = Nspace[rN]
            # free space now that v is captured
            if rN not in Cset:
                del Nspace[rN]

            cS = row2col[rN]
            for cN in cS:
                E = self.Espace[rN, cN]
                prev = Nspace.get(cN, None)
                if v is None:
                    addin = E
                else:
                    addin = v @ E
                if prev is None:
                    Nspace[cN] = addin
                    Nnum[cN] = 1
                else:
                    # TODO, could make this in-place
                    Nspace[cN] = prev + addin
                    Nnum[cN] += 1
                # this condition means that the row node has been fully filled by
                # all col nodes
                if Nnum[cN] == len(col2row[cN]):
                    row_stack.append(cN)

        assert(set(Nspace.keys()) == Cset)

        # ops.append(OpComp("N_edge", node, (stk.key_edge(node, cN),)))
        # ops.append(
        #     OpComp("N_sum", node, (stk.key_edge(node, cN), cN))
        # )
        # ops.append(OpComp("N_ret", R, ()))
        return {k: self.typemap_fr(v) for k, v in Nspace.items()}

    def inverse_derivative(self, Cmap, Rset, Dmap):
        """
        Calculate the derivative of the matrix inverse along the given derivative values.

        this calculates Sum_i Dmap_i @ (M^-1)'_i by using Dmap_i @ (M^-1)'_i = Dmap_i * (M^-1 @ M'_i @ M^-1). Notably the value of M appears twice and "self" here is the second one. The first M^-1 is supplied by Cmap.

        This method of calling allow for the first and second M^-1 to come from separate M evaluations. In particular,
        it allows a DC computation to be applied first.

        Cmap is a column mapping from a previous evaluation of
        inverse_col(Cmap, Rset=any, derivative=True)
        where Cmap is the original driving vector and Rset can be anything since it is augmented
        appropriately by derivatives=True

        Dmap must be a dictionary mapping edge pair-tuples or edge value-names to matrices.
        """

        Dval2 = {}
        for D, Dval in Dmap:
            if isinstance('D', str):
                edges = self.edges_rev[D]
                for edge in edges:
                    Dval2[edge] = Dval
            else:
                Dval2[edge] = Dval

        Cmap2 = {}
        # now apply matrix multiplication through Dval2
        for (DR, DC), DV in Dval2.items():
            mul = DV @ Cmap[DC]
            prev = Cmap2.setdefault(DR, mul)
            if mul is not prev:
                Cmap2[DR] = prev + mul

        return self.inverse_col(Rset, Cmap2)



    def inverse_single(self, rN, cN):
        """
        This computes the matrix element for an inverse from C to R
        """
        return self.inverse_col([rN], {cN: None})[rN]
        # return self.inverse_row({rN: None}, {cN})[cN]

    # #################################
    # ############### serialization

    @classmethod
    def from_yaml(cls, y, **kwargs):
        if isinstance(y, str):
            y = yaml.safe_load(y)
        oplistE = cls.convert_yamlpy2oplistE(y['oplistE'])
        row2col = cls.convert_yamlpy2row2col(y['row2col'])
        edges = cls.convert_yamlpy2edges(y['edges'])

        col2row = defaultdict(set)
        for rN, cS in row2col.items():
            for cN in cS:
                col2row[cN].add(rN)
        col2row = dict(col2row)

        return cls(
            oplistE=oplistE,
            edges=edges,
            row2col=row2col,
            col2row=col2row,
            **kwargs,
        )

    def convert_self2yamlpy(self):
        yamlpy_oplistE = self.convert_oplistE2yamlpy()
        yamlpy_edges = self.convert_edges2yamlpy()
        yamlpy_row2col = self.convert_row2col2yamlpy()

        return dict(
            oplistE = yamlpy_oplistE,
            edges = yamlpy_edges,
            row2col = yamlpy_row2col,
        )

    def convert_self2yamlstr(self):
        return yaml.safe_dump(
            self.convert_self2yamlpy(),
            default_flow_style=None,
        )

    def convert_oplistE2yamlpy(self):
        oplistE_yamlpy = []

        for op in self.oplistE:
            n, targ, args = op
            args = [yamlstr_convert(a) for a in args]
            odict = dict(op=n, targ=yamlstr_convert(targ))
            if args:
                odict['args'] = args
            oplistE_yamlpy.append(odict)

        return oplistE_yamlpy

    @classmethod
    def convert_yamlpy2oplistE(cls, yamlpy):
        oplist_conv = []
        for op in yamlpy:
            oplist_conv.append(
                SFLU.OpComp(
                    op['op'],
                    yamlstr_convert_rev(op['targ']),
                    tuple(yamlstr_convert_rev(arg) for arg in op.get('args', ())),
                )
            )
        return oplist_conv

    def convert_row2col2yamlpy(self):
        row2col = {}

        for rN, cS in self.row2col.items():
            row2col[yamlstr_convert(rN)] = [yamlstr_convert(cN) for cN in cS]

        return row2col

    @classmethod
    def convert_yamlpy2row2col(cls, yamlpy):
        row2col = {}

        for rN, cS in yamlpy.items():
            row2col[yamlstr_convert_rev(rN)] = set(yamlstr_convert_rev(cN) for cN in cS)

        return row2col

    def convert_edges2yamlpy(self):
        edges = {}

        for k, v in self.edges.items():
            edges[yamlstr_convert(k)] = v

        return edges

    @classmethod
    def convert_yamlpy2edges(cls, yamlpy):
        edges = {}

        for k, v in yamlpy.items():
            edges[yamlstr_convert_rev(k)] = normalize_list2tuple(v)

        return edges

    def convert_oplistE2yamlstr(self):
        oplistE_yaml = yaml.safe_dump(
            self.convert_oplistE2yamlpy(),
            default_flow_style=None,
        )
        return oplistE_yaml

    @classmethod
    def convert_yamlstr2oplistE(cls, s):
        oplistE_yamlpy = yaml.safe_load(s)
        return cls.convert_yamlpy2oplistE(oplistE_yamlpy)
