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
import itertools
import yaml
import re

from collections import defaultdict, namedtuple

from wield.utilities.np import matrix_stack

from ..import string_tuple_keys as stk

from . import SFLUcompute
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import sympy as sp
except ImportError:
    sp = None

from .conversions import (
    yamlstr_convert,
    yamlstr_convert_rev,
    normalize_list2tuple,
)

Op = namedtuple("Op", ("op", "args"))
OpComp = namedtuple("OpComp", ("op", "targ", "args"))


class SFLU(object):
    def __init__(
        self,
        edges,
        derivatives=[],
        reduce_list=[],
        inputs=None,
        outputs=None,
        graph=False,
    ):
        """
        This takes a dictionary of edges which are (row, col) tuples, matched to a value name.

        The value name will be mapped into the Espace

        derivatives: this argument is a list of either edge pair-tuples or of edge labels. It
        establishes the list of testpoint and excitation inputs and outputs required to take
        derivatives.
        """
        if graph:
            self.G = nx.DiGraph()
        else:
            self.G = None

        self.inputs_init = inputs
        self.outputs_init = outputs
        self.edges_init = edges
        self.reduce_list = reduce_list
        # this takes a column and provides the row edges
        col2row = defaultdict(set)
        # this takes a row and provides the column edges
        row2col = defaultdict(set)

        nodes = set()

        # this second set are the row and col edge sets that are cycle-free
        col2row_cf = defaultdict(set)
        row2col_cf = defaultdict(set)

        edges2 = dict()
        # these are the original edges, augmented by derivative input/outputs
        # and with mapped input/outputs
        edges_augmented = dict()
        edges2_reverse = defaultdict(set)

        def add_edge(R, C, E):
            edge = stk.key_edge(R, C)
            edges2[edge] = E
            edges_augmented[edge] = E
            edges2_reverse[E].add(edge)
            col2row[C].add(R)
            row2col[R].add(C)
            nodes.add(R)
            nodes.add(C)

        # dress up add edge if the graph is around
        if self.G is not None:
            add_edge_prev = add_edge

            def add_edge(R, C, E):
                add_edge_prev(R, C, E)
                self.G.add_edge(C, R, label_default=to_label(E))

        # now loop through the edges
        for (R, C), E in edges.items():
            # normalize computation-based edges
            E = normalize_list2tuple(E)
            R = stk.key_map(R)
            C = stk.key_map(C)
            add_edge(R, C, E)


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

        self.derivatives = derivatives

        def add_derivative(R, C):
            if R not in inputs:
                Ri = stk.key_join(R, 'Di')
                add_edge(R, Ri, '1')
                inputs.add(Ri)
                inputs_.add(Ri)
            else:
                raise NotImplementedError("Currently does not support derivatives on edges of input nodes")
            if C not in outputs:
                Co = stk.key_join(C, 'Do')
                outputs.add(Co)
                outputs_.add(Co)
                add_edge(Co, C, '1')
            else:
                raise NotImplementedError("Currently does not support derivatives on edges of output nodes")
        
            if self.G is not None:
                self.G.nodes[Ri]['pos'] = None
                self.G.nodes[Co]['pos'] = None
                self.G.edges[Ri, R]['suppress'] = True
                self.G.edges[C, Co]['suppress'] = True

        for D in derivatives:
            if isinstance(D, str):
                for R, C in edges2_reverse[D]:
                    add_derivative(R, C)
            else:
                add_derivative(R, C)

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

        # label all nodes
        if self.G is not None:
            for n in itertools.chain(nodes, inputs, outputs):
                self.G.nodes[n]['label_default'] = to_label(n)

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
        self.edges_augmented = edges_augmented

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
                    x, y = pos
                    self.G.nodes[n]['pos'] = float(x), float(y)
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
                    x, y = p
                    self.G.nodes[n]['pos'] = float(x), float(y)
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
            self.G.nodes[iN]['angle'] = -135

            # eset = self.col2row_cf[iN]
            # for rN in eset:
            #     self.G.edges[rN, iN]['bend'] = 0

        def key(oN):
            oS = self.row2col_cf[oN]
            return tuple(sorted([reducedU2.index(c) for c in oS]))

        Y_ = Y
        for oN in sorted(self.outputs, key=key):
            self.graph_nodes_pos({
                oN: (rX, Y_),
            })
            self.G.nodes[oN]['angle'] = -45
            Y_ += dY

        return

    def convert_self2yamlpy(self):
        s = dict()
        s['edges'] = {yamlstr_convert(stk.key_edge(*edge)): v for edge, v in self.edges_init.items()}
        s['derivatives'] = self.derivatives
        s['reduce_list'] = self.reduce_list
        if self.inputs_init is not None:
            s['inputs'] = list(self.inputs_init)
        if self.outputs_init is not None:
            s['outputs'] = list(self.outputs_init)
        if self.G:
            G_nodes = s['G_nodes'] = dict()
            G_edges = s['G_edges'] = dict()
            for node, ndict in self.G.nodes.items():
                ndict = dict(ndict)
                ndict.pop('label_default', None)
                if ndict:
                    G_nodes[yamlstr_convert(node)] = ndict
            for edge, edict in self.G.edges.items():
                edict = dict(edict)
                edict.pop('label_default', None)
                if edict:
                    G_edges[yamlstr_convert(stk.key_edge(*edge))] = edict
        return s

    def convert_self2yamlstr(self):
        d = self.convert_self2yamlpy()
        d2 = dict()

        def transfer(name):
            val = d.pop(name, None)
            if val is not None:
                d2[name] = val

        transfer('reduce_list')
        transfer('G_nodes')
        transfer('G_edges')

        s1 = yaml.safe_dump(
            d, default_flow_style=False,
            sort_keys=False,
        )
        if d2:
            s2 = yaml.safe_dump(
                d2,
                default_flow_style=None,
                sort_keys=False,
            )
        else:
            s2 = ''
        return s1 + s2

    @classmethod
    def convert_yamlstr2self(cls, yamlstr):
        yamlpy = yaml.safe_load(yamlstr)
        kw = dict(
            edges={yamlstr_convert_rev(edge): v for edge, v in yamlpy['edges'].items()},
            derivatives=yamlpy['derivatives'],
        )

        inputs = yamlpy.get('inputs', None)
        if inputs is not None:
            kw['inputs'] = inputs

        outputs = yamlpy.get('outputs', None)
        if outputs is not None:
            kw['outputs'] = outputs

        reduce_list = yamlpy.get('reduce_list', None)
        if reduce_list is not None:
            kw['reduce_list'] = reduce_list

        if 'G_nodes' in yamlpy:
            kw['graph'] = True
        self = cls(
            **kw
        )
        if self.G is not None:
            for node, ndict in yamlpy['G_nodes'].items():
                self.G.nodes[node].update(ndict)
            for edge, edict in yamlpy['G_edges'].items():
                self.G.edges[yamlstr_convert_rev(edge)].update(edict)
        return self

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

    def reduce_auto(self):
        self.reduce(*self.reduce_list)
        self.reduce(*self.nodes)
        return

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
                self.G.nodes[NsfA]['label_default'] = to_label(NsfA)
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
                self.G.nodes[NsfB]['label_default'] = to_label(NsfB)
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
                if R not in self.outputs:
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
                if C not in self.inputs:
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
        return SFLUcompute.SFLUCompute(
            oplistE=self.oplistE,
            edges=self.edges_augmented,
            row2col=dict(self.row2col_cf),
            col2row=dict(self.col2row_cf),
            **kwargs
        )


def to_label(val):
    if not val:
        return ""
    if sp is not None:
        return '$' + sp.latex(sp.var(str(val))) + '$'
    return val
