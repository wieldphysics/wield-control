#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import yaml
import re

from collections import defaultdict, namedtuple

from wavestate.utilities.np import matrix_stack

from ..import string_tuple_keys as stk
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import sympy as sp
except ImportError:
    sp = None


Op = namedtuple("Op", ("op", "args"))
OpComp = namedtuple("OpComp", ("op", "targ", "args"))


class SFLU(object):
    def __init__(
        self,
        edges,
        inputs=None,
        outputs=None,
        graph=False,
    ):
        """
        This takes a dictionary of edges which are (row, col) tuples, matched to a value name.

        The value name will be mapped into the Espace
        """
        if graph:
            self.G = nx.DiGraph()
        else:
            self.G = None

        # this takes a column and provides the row edges
        col2row = defaultdict(set)
        # this takes a row and provides the column edges
        row2col = defaultdict(set)

        nodes = set()

        # this second set are the row and col edge sets that are cycle-free
        col2row_cf = defaultdict(set)
        row2col_cf = defaultdict(set)

        edges2 = dict()
        edges_original = dict()

        for (R, C), E in edges.items():
            # normalize computation-based edges
            E = normalize_list2tuple(E)

            R = stk.key_map(R)
            C = stk.key_map(C)
            edges2[stk.key_edge(R, C)] = E
            edges_original[stk.key_edge(R, C)] = E
            col2row[C].add(R)
            row2col[R].add(C)
            nodes.add(R)
            nodes.add(C)

        if self.G is not None:
            for (R, C), E in edges2.items():
                self.G.add_edge(C, R, label=to_label(E))
            for n in nodes:
                self.G.nodes[n]['label'] = to_label(n)
        # print('col2row', col2row)
        # print('row2col', row2col)

        # TODO: need to remove missing inputs and output nodes from
        # the node list
        # ONLY matters if inputs/outputs are specified

        inputs_ = set()
        for rN in nodes:
            cS = row2col.get(rN, None)
            if cS is None or len(cS) == 0:
                inputs_.add(rN)
            elif len(cS) == 1 and rN in cS:
                inputs_.add(rN)

        outputs_ = set()
        for cN in nodes:
            rS = col2row.get(cN, None)
            if rS is None or len(rS) == 0:
                outputs_.add(cN)
            elif len(rS) == 1 and cN in rS:
                outputs_.add(cN)

        if inputs is not None:
            inputs = set(inputs)
            assert(inputs.issubset(inputs_))
        else:
            inputs = inputs_

        if outputs is not None:
            outputs = set(outputs)
            assert(outputs.issubset(outputs_))
        else:
            outputs = outputs_
        outputs = set(outputs)
        assert(outputs.issubset(outputs_))

        for iN in inputs_:
            cS = row2col[iN]
            do_move = (iN in inputs)
            if len(cS) == 1:
                assert iN in cS
                if do_move:
                    row2col_cf[iN].update(cS)
            else:
                assert len(cS) == 0
            del row2col[iN]

            rS = col2row[iN]
            dS = rS  # rS.intersection(outputs)
            if do_move:
                col2row_cf[iN].update(dS)
            for rN in dS:
                row2col[rN].remove(iN)
                if do_move:
                    row2col_cf[rN].add(iN)

            rS.difference_update(dS)
            nodes.remove(iN)
            if self.G is not None and not do_move:
                self.G.remove_node(iN)

        for oN in outputs_:
            do_move = (oN in outputs)
            rS = col2row[oN]
            if len(rS) == 1:
                assert oN in rS
                if do_move:
                    row2col_cf[oN].update(rS)
            else:
                assert len(rS) == 0
            del col2row[oN]

            cS = row2col[oN]
            dS = cS  # cS.intersection(inputs)
            if do_move:
                row2col_cf[oN].update(dS)
            for cN in dS:
                col2row[cN].remove(oN)
                if do_move:
                    col2row_cf[cN].add(oN)
            cS.difference_update(dS)
            nodes.remove(oN)
            if self.G is not None and not do_move:
                self.G.remove_node(oN)

        self.col2row = col2row
        self.row2col = row2col
        self.dropped = (outputs_ - outputs) | (inputs_ - inputs)

        self.col2row_cf = col2row_cf
        self.row2col_cf = row2col_cf

        def check(col2row, row2col):
            for cN, rS in col2row.items():
                for rN in rS:
                    assert cN in row2col[rN]
            for rN, cS in row2col.items():
                for cN in cS:
                    assert rN in col2row[cN]

        check(col2row, row2col)
        check(col2row_cf, row2col_cf)

        self.edges = edges2
        #  the original edges
        self.edges_original = edges_original

        self.nodes = nodes
        self.reduced = []
        self.reducedL = []
        self.reducedU = []

        self.inputs = inputs
        self.outputs = outputs

        # the series of operations to act on the edge space during computation
        self.oplistE = []
        return

    def graph_labels(self, labels):
        pos2 = {}
        for n, l in labels.items():
            n = stk.key_map(n)
            pos2[n] = l
        nx.set_node_attributes(self.G, pos2, 'label')
        return

    def graph_positions(self, pos):
        pos2 = {}
        for n, p in pos.items():
            n = stk.key_map(n)
            pos2[n] = p
        nx.set_node_attributes(self.G, pos2, 'pos')
        return

    def graph_nodes_repr(self):
        strs = []
        for n in self.G.nodes:
            strs.append(str(n))
        return strs

    def graph_nodes_pos(self, pos, *nodes, match=True):
        """
        Assigns the position keyword to nodes.

        The first argument "pos" is a tuple of (x, y) locations.
        """
        if nodes:
            # assign pos to each node in nodes
            for n in nodes:
                if n in self.dropped:
                    continue
                n = stk.key_map(n)
                try:
                    self.G.nodes[n]['pos'] = pos
                except KeyError:
                    if match:
                        raise
        else:
            # assumes it is a dictionary
            for n, p in pos.items():
                n = stk.key_map(n)
                if n in self.dropped:
                    continue
                try:
                    self.G.nodes[n]['pos'] = p
                except KeyError:
                    if match:
                        raise
        return

    def graph_nodes_posX(self, posX, *nodes):
        for n in nodes:
            if n in self.dropped:
                continue
            n = stk.key_map(n)
            pos = self.G.nodes[n].get('pos', (None, None))
            self.G.nodes[n]['pos'] = (posX, pos[1])

    def graph_nodes_posY(self, posY, *nodes):
        for n in nodes:
            if n in self.dropped:
                continue
            n = stk.key_map(n)
            pos = self.G.nodes[n].get('pos', (None, None))
            self.G.nodes[n]['pos'] = (pos[0], posY)

    def graph_nodes_pos_get(self, *nodes):
        if not nodes:
            pos = nx.get_node_attributes(self.G, 'pos')
            pos2 = {}
            for k, p in pos.items():
                pos2[str(k)] = p
            return pos2

        for n in nodes:
            if n in self.dropped:
                continue
            n2 = stk.key_map(n)
            p = self.G.nodes[n2].get('pos', None)
            if p is None:
                continue
            pos2[n] = p
            return pos2

    _G_reduce_lX_rX_Y_dY = None

    def graph_reduce_auto_pos(self, lX, rX, Y, dY):
        self._G_reduce_lX_rX_Y_dY = (lX, rX, Y, dY)
        return

    def graph_reduce_auto_pos_io(self, lX, rX, Y, dY):
        # prevent issue where inputs are connected directly to outputs
        reducedL2 = list(self.outputs) + list(self.reducedL)
        reducedU2 = list(self.inputs) + list(self.reducedU)

        def key(iN):
            iS = self.col2row_cf[iN]
            return tuple(sorted([reducedL2.index(r) for r in iS]))
        Y_ = Y
        for iN in sorted(self.inputs, key=key):
            self.graph_nodes_pos({
                iN: (lX, Y_),
            })
            Y_ += dY

        def key(oN):
            oS = self.row2col_cf[oN]
            return tuple(sorted([reducedU2.index(c) for c in oS]))
        Y_ = Y
        for oN in sorted(self.outputs, key=key):
            self.graph_nodes_pos({
                oN: (rX, Y_),
            })
            Y_ += dY
        return

    def invertE(self, E):
        return Op("invert", E)

    def addE(self, *Es):
        flat = []
        for E in Es:
            if isinstance(E, tuple):
                if E[0] == "add":
                    flat.extend(E[1:])
                else:
                    flat.append(E)
            else:
                flat.append(E)
        return Op("add", tuple(flat))

    def mulE(self, *Es):
        flat = []
        for E in Es:
            if isinstance(E, tuple):
                if E[0] == "mul":
                    flat.extend(E[1:])
                else:
                    flat.append(E)
            elif E == 1:
                # don't include a unity in a mul
                pass
            else:
                flat.append(E)
        if len(flat) == 1:
            return flat[0]
        elif len(flat) == 0:
            return 0
        else:
            return Op("mul", tuple(flat))

    def reduce(self, *nodes):
        for node in nodes:
            self.reduce_single(node)
        return

    def reduce_single(self, node):
        Nsf = stk.key_map(node)

        # the following two if statements
        # determine if one of the split L/U nodes
        # is not necessary since it doesn't reach an output
        NsfB = stk.key_join("U", Nsf)
        NsfA = stk.key_join("L", Nsf)
        if self.col2row_cf[Nsf]:
            NsfB_needed = True
        else:
            NsfB_needed = False

        if self.row2col_cf[Nsf]:
            NsfA_needed = True
        else:
            NsfA_needed = False

        selfE = self.edges.get((Nsf, Nsf), None)

        CLG = self.invertE(selfE)

        if selfE is not None:
            # remove the self edge before the simplification stage
            self.col2row[Nsf].remove(Nsf)
            self.row2col[Nsf].remove(Nsf)
            del self.edges[Nsf, Nsf]

        # add the direct connection
        if NsfA_needed and NsfB_needed:
            self.edges[NsfB, NsfA] = CLG
            self.col2row_cf[NsfA].add(NsfB)
            self.row2col_cf[NsfB].add(NsfA)
            if self.G is not None:
                self.G.add_edge(NsfA, NsfB)
                self.G.edges[NsfA, NsfB]['bend'] = 0

        # save_self_edge indicates if the self edge could be
        # deleted at the end
        delete_self_edge = False
        if (
                (self.row2col[Nsf] or self.col2row[Nsf])
                or (self.row2col_cf[Nsf] or self.col2row_cf[Nsf])
        ):
            if not (NsfA_needed and NsfB_needed):
                delete_self_edge = True
            if selfE is not None:
                self.oplistE.append(
                    OpComp(
                        "E_CLG",
                        stk.key_edge(NsfB, NsfA),
                        (stk.key_edge(Nsf, Nsf),),
                    )
                )
            else:
                self.oplistE.append(
                    # this one has a strange arg type of a node-op
                    # it indicates a default self-edge operation
                    OpComp("E_CLGd", stk.key_edge(NsfB, NsfA), (Nsf,))
                )

        # process all of the internal edges from the main graph to itself
        for R in self.col2row[Nsf]:
            edgeR = self.edges[R, Nsf]
            for C in self.row2col[Nsf]:
                edgeC = self.edges[Nsf, C]

                ACedge = self.edges.get((R, C), None)
                if ACedge is not None:
                    self.edges[(R, C)] = self.addE(self.mulE(edgeR, CLG, edgeC), ACedge)
                    self.oplistE.append(
                        OpComp(
                            "E_mul3add",
                            stk.key_edge(R, C),
                            (
                                stk.key_edge(R, Nsf),
                                stk.key_edge(NsfB, NsfA),
                                stk.key_edge(Nsf, C),
                                stk.key_edge(R, C),
                            ),
                        )
                    )
                else:
                    self.edges[(R, C)] = self.mulE(edgeR, CLG, edgeC)
                    self.oplistE.append(
                        OpComp(
                            "E_mul3",
                            stk.key_edge(R, C),
                            (
                                stk.key_edge(R, Nsf),
                                stk.key_edge(NsfB, NsfA),
                                stk.key_edge(Nsf, C),
                            ),
                        )
                    )

                self.col2row[C].add(R)
                self.row2col[R].add(C)
                if self.G is not None:
                    self.G.add_edge(C, R)
                    self.G.edges[C, R]['bend'] = -10
                    if C == R:
                        self.G.edges[C, R]['suppress'] = False

        # edges from the cycle-free nodes back into the main graph
        # (the following two for-loops)

        if NsfA_needed:
            for R in self.col2row[Nsf]:
                edge = self.edges.pop((R, Nsf))
                self.edges[R, NsfA] = self.mulE(edge, CLG)
                self.oplistE.append(
                    OpComp(
                        "E_mul2",
                        stk.key_edge(R, NsfA),
                        (stk.key_edge(R, Nsf), stk.key_edge(NsfB, NsfA)),
                    )
                )
                self.oplistE.append(OpComp("E_del", stk.key_edge(R, Nsf), ()))
                self.col2row_cf[NsfA].add(R)
                self.row2col_cf[R].add(NsfA)
                self.row2col[R].remove(Nsf)

                if self.G is not None:
                    self.G.add_edge(NsfA, R, type='no_cycle')
            if self.G is not None:
                self.G.nodes[NsfA]['label'] = to_label(NsfA)
        else:
            for R in self.col2row[Nsf]:
                edge = self.edges.pop((R, Nsf))
                self.row2col[R].remove(Nsf)
        del self.col2row[Nsf]

        if NsfB_needed:
            for C in self.row2col[Nsf]:
                edge = self.edges.pop((Nsf, C))
                self.edges[NsfB, C] = self.mulE(CLG, edge)
                self.oplistE.append(
                    OpComp(
                        "E_mul2",
                        stk.key_edge(NsfB, C),
                        (
                            stk.key_edge(NsfB, NsfA),
                            stk.key_edge(Nsf, C),
                        ),
                    )
                )
                self.oplistE.append(OpComp("E_del", stk.key_edge(Nsf, C), ()))
                self.col2row_cf[C].add(NsfB)
                self.row2col_cf[NsfB].add(C)
                self.col2row[C].remove(Nsf)

                if self.G is not None:
                    self.G.add_edge(C, NsfB, type='no_cycle')
            if self.G is not None:
                self.G.nodes[NsfB]['label'] = to_label(NsfB)
                self.G.nodes[NsfB]['angle'] = -135
        else:
            for C in self.row2col[Nsf]:
                edge = self.edges.pop((Nsf, C))
                self.col2row[C].remove(Nsf)
        del self.row2col[Nsf]

        # edges just between the cycle-free elements
        # (the following two for-loops)
        for R in self.col2row_cf[Nsf]:
            edge = self.edges.pop((R, Nsf))
            self.edges[R, NsfB] = edge
            self.oplistE.append(
                OpComp(
                    "E_assign",
                    stk.key_edge(R, NsfB),
                    (stk.key_edge(R, Nsf),),
                )
            )
            self.oplistE.append(OpComp("E_del", stk.key_edge(R, Nsf), ()))
            self.col2row_cf[NsfB].add(R)
            self.row2col_cf[R].add(NsfB)
            self.row2col_cf[R].remove(Nsf)
            if self.G is not None:
                self.G.add_edge(NsfB, R)
                self.G.edges[NsfB, R]['bend'] = 25
        del self.col2row_cf[Nsf]

        for C in self.row2col_cf[Nsf]:
            edge = self.edges.pop((Nsf, C))
            self.edges[NsfA, C] = edge
            self.oplistE.append(
                OpComp(
                    "E_assign",
                    stk.key_edge(NsfA, C),
                    (stk.key_edge(Nsf, C),),
                )
            )
            self.oplistE.append(OpComp("E_del", stk.key_edge(Nsf, C), ()))
            self.col2row_cf[C].add(NsfA)
            self.row2col_cf[NsfA].add(C)
            self.col2row_cf[C].remove(Nsf)

            if self.G is not None:
                self.G.add_edge(C, NsfA)
                self.G.edges[C, NsfA]['bend'] = 25
        del self.row2col_cf[Nsf]

        # check save_self_edge and add the delete operator
        if delete_self_edge:
            self.oplistE.append(OpComp("E_del", stk.key_edge(NsfB, NsfA), ()))

        self.nodes.remove(Nsf)

        self.reduced.append(Nsf)
        if NsfA_needed:
            self.reducedL.append(NsfA)
        if NsfB_needed:
            self.reducedU.append(NsfB)

        if self.G is not None:
            self.G.remove_node(Nsf)
            if self._G_reduce_lX_rX_Y_dY is not None:
                lX, rX, Y, dY = self._G_reduce_lX_rX_Y_dY
                if NsfA_needed:
                    self.graph_nodes_pos({
                        NsfA: (lX, Y),
                    })
                if NsfB_needed:
                    self.graph_nodes_pos({
                        NsfB: (rX, Y),
                    })
                self._G_reduce_lX_rX_Y_dY = (lX, rX, Y + dY, dY)

        return True

    def computer(self, **kwargs):
        return SFLUCompute(
            oplistE = self.oplistE,
            edges   = self.edges_original,
            row2col = dict(self.row2col_cf),
            col2row = dict(self.col2row_cf),
            **kwargs
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
    ):
        self.oplistE     = oplistE
        self.edges       = edges
        self.row2col     = row2col
        self.col2row     = col2row

        if typemap_to is None:
            typemap_to = self.typemap_to_default
        if typemap_fr is None:
            typemap_fr = self.typemap_fr_default
        self.typemap_to = typemap_to
        self.typemap_fr = typemap_fr
        self.eye = self.typemap_to(1)
        return

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

    def edge_map(self, edge_map, default = None):
        def edge_compute(ev):
            if isinstance(ev, str):
                if ev[0] == '-':
                    ev = ev[1:]
                    if default is not False:
                        return self.typemap_to(-edge_map.setdefault(ev, default))
                    else:
                        return self.typemap_to(-edge_map[ev])
                else:
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
        Espace = self.edge_map(edge_map, default = False)

        for op in self.oplistE:
            # print(op)
            if op.op == "E_CLG":
                (arg,) = op.args
                E = Espace[arg]
                # assert E.shape[-1] == E.shape[-2]
                # I = np.eye(E.shape[-1])
                # I = I.reshape((1,) * (len(E.shape) - 2) + (E.shape[-2:]))

                E2 = (self.eye - E)**-1

                Espace[op.targ] = E2
                # print("CLG: ", op.targ, E2)

            elif op.op == "E_CLGd":
                # (arg,) = op.args
                # size = self.nodesizes.get(arg, self.defaultsize)
                # I = np.eye(size)
                Espace[op.targ] = self.eye

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

    def inverse_col(self, Rset, Cmap):
        """
        This computes the matrix element for an inverse from C to R

        TODO, the algorithm could/should use some work. Should make a copy of col2row
        then deplete it
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

    def inverse_row(self, Rmap, Cset):
        """
        This computes the matrix element for an inverse from C to R

        TODO, the algorithm could/should use some work. Should make a copy of col2row
        then deplete it
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
                OpComp(
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


RE_VEC = re.compile(r"\((.*)<(.*)\)")


def yamlstr_convert(a):
    # TODO, make this a little less fragile for generalized edges
    # also check that node names don't look like edges
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
