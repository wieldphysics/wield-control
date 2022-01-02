#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
from collections import defaultdict, namedtuple
from ..statespace import tupleize

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
        row = defaultdict(set)
        # this takes a row and provides the column edges
        col = defaultdict(set)

        nodes = set()

        # this second set are the row and col edge sets that are cycle-free
        row2 = defaultdict(set)
        col2 = defaultdict(set)

        edges2 = dict()
        edges_original = dict()

        for (R, C), E in edges.items():
            R = tupleize.tupleize(R)
            C = tupleize.tupleize(C)
            edges2[tupleize.EdgeTuple(R, C)] = E
            edges_original[tupleize.EdgeTuple(R, C)] = E
            row[C].add(R)
            col[R].add(C)
            nodes.add(R)
            nodes.add(C)

        if self.G is not None:
            def to_label(val):
                return '$' + sp.latex(sp.var(str(val))) + '$'
            for (R, C), E in edges2.items():
                self.G.add_edge(C, R, label=to_label(E))
            for n in nodes:
                print('tuplized: ', str(n))
                self.G.nodes[n]['label'] = to_label(n)
        # print('row', row)
        # print('col', col)

        if inputs is None:
            inputs = set()
            for rN in nodes:
                cS = col.get(rN, None)
                if cS is None or len(cS) == 0:
                    inputs.add(rN)
                elif len(cS) == 1 and rN in cS:
                    inputs.add(rN)

        if outputs is None:
            outputs = set()
            for cN in nodes:
                rS = row.get(cN, None)
                if rS is None or len(rS) == 0:
                    outputs.add(cN)
                elif len(rS) == 1 and cN in rS:
                    outputs.add(cN)

        for iN in inputs:
            cS = col[iN]
            if len(cS) == 1:
                assert iN in cS
                col2[iN].update(cS)
            else:
                assert len(cS) == 0
            del col[iN]

            rS = row[iN]
            dS = rS  # rS.intersection(outputs)
            row2[iN].update(dS)
            for rN in dS:
                col[rN].remove(iN)
                col2[rN].add(iN)

            rS.difference_update(dS)

        for oN in outputs:
            rS = row[oN]
            if len(rS) == 1:
                assert oN in rS
                col2[oN].update(rS)
            else:
                assert len(rS) == 0
            del row[oN]

            cS = col[oN]
            dS = cS  # cS.intersection(inputs)
            col2[oN].update(dS)
            for cN in dS:
                row[cN].remove(oN)
                row2[cN].add(oN)
            cS.difference_update(dS)

        self.row = row
        self.col = col

        self.row2 = row2
        self.col2 = col2

        def check(row, col):
            for cN, rS in row.items():
                for rN in rS:
                    assert cN in col[rN]
            for rN, cS in col.items():
                for cN in cS:
                    assert rN in row[cN]

        check(row, col)
        check(row2, col2)

        self.edges = edges2
        #  the original edges
        self.edges_original = edges_original

        self.nodes = nodes

        self.inputs = inputs
        self.outputs = outputs

        # the series of operations to act on the edge space during computation
        self.oplistE = []
        return

    def graph_labels(self, labels):
        pos2 = {}
        for n, l in labels.items():
            n = tupleize.tupleize(n)
            pos2[n] = l
        nx.set_node_attributes(self.G, pos2, 'label')
        return

    def graph_positions(self, pos):
        pos2 = {}
        for n, p in pos.items():
            n = tupleize.tupleize(n)
            pos2[n] = p
        nx.set_node_attributes(self.G, pos2, 'pos')
        return

    def graph_nodes_repr(self):
        strs = []
        for n in self.G.nodes:
            strs.append(str(n))
        return strs

    def graph_nodes_pos(self, pos, *nodes):
        if nodes:
            # assign pos to each node in nodes
            for n in nodes:
                n = tupleize.tupleize(n)
                self.G.nodes[n]['pos'] = pos
        else:
            # assumes it is a dictionary
            for n, p in pos.items():
                n = tupleize.tupleize(n)
                self.G.nodes[n]['pos'] = p
        return

    def graph_nodes_posX(self, posX, *nodes):
        for n in nodes:
            n = tupleize.tupleize(n)
            pos = self.G.nodes[n].get('pos', (None, None))
            self.G.nodes[n]['pos'] = (posX, pos[1])

    def graph_nodes_posY(self, posY, *nodes):
        for n in nodes:
            n = tupleize.tupleize(n)
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
            n2 = tupleize.tupleize(n)
            p = self.G.nodes[n2].get('pos', None)
            if p is None:
                continue
            pos2[n] = p
            return pos2

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

    def reduce(self, node):
        Nsf = tupleize.tupleize(node)
        NsfB = tupleize.tupleize("U") + Nsf
        NsfA = tupleize.tupleize("L") + Nsf

        selfE = self.edges.get((Nsf, Nsf), None)

        CLG = self.invertE(selfE)

        if selfE is not None:
            # remove the self edge before the simplification stage
            self.row[Nsf].remove(Nsf)
            self.col[Nsf].remove(Nsf)
            del self.edges[Nsf, Nsf]

        # add the direct connection
        self.edges[NsfB, NsfA] = CLG
        self.row2[NsfA].add(NsfB)
        self.col2[NsfB].add(NsfA)
        if self.G is not None:
            self.G.add_edge(NsfA, NsfB)

        if selfE is not None:
            self.oplistE.append(
                OpComp(
                    "E_CLG",
                    tupleize.EdgeTuple(NsfB, NsfA),
                    (tupleize.EdgeTuple(Nsf, Nsf),),
                )
            )
        else:
            self.oplistE.append(
                # this one has a strange arg type of a node
                OpComp("E_CLGd", tupleize.EdgeTuple(NsfB, NsfA), (Nsf,))
            )

        # process all of the internal edges
        for R in self.row[Nsf]:
            edgeR = self.edges[R, Nsf]
            for C in self.col[Nsf]:
                edgeC = self.edges[Nsf, C]

                ACedge = self.edges.get((R, C), None)
                if ACedge is not None:
                    self.edges[(R, C)] = self.addE(self.mulE(edgeR, CLG, edgeC), ACedge)
                    self.oplistE.append(
                        OpComp(
                            "E_mul3add",
                            tupleize.EdgeTuple(R, C),
                            (
                                tupleize.EdgeTuple(R, Nsf),
                                tupleize.EdgeTuple(NsfB, NsfA),
                                tupleize.EdgeTuple(Nsf, C),
                                tupleize.EdgeTuple(R, C),
                            ),
                        )
                    )
                else:
                    self.edges[(R, C)] = self.mulE(edgeR, CLG, edgeC)
                    self.oplistE.append(
                        OpComp(
                            "E_mul3",
                            tupleize.EdgeTuple(R, C),
                            (
                                tupleize.EdgeTuple(R, Nsf),
                                tupleize.EdgeTuple(NsfB, NsfA),
                                tupleize.EdgeTuple(Nsf, C),
                            ),
                        )
                    )

                self.row[C].add(R)
                self.col[R].add(C)
                if self.G is not None:
                    self.G.add_edge(C, R)

        # now process all of the no_cycle edges
        for R in self.row[Nsf]:
            edge = self.edges.pop((R, Nsf))
            self.edges[R, NsfA] = self.mulE(edge, CLG)
            self.oplistE.append(
                OpComp(
                    "E_mul2",
                    tupleize.EdgeTuple(R, NsfA),
                    (tupleize.EdgeTuple(R, Nsf), tupleize.EdgeTuple(NsfB, NsfA)),
                )
            )
            self.oplistE.append(OpComp("E_del", tupleize.EdgeTuple(R, Nsf), ()))
            self.row2[NsfA].add(R)
            self.col2[R].add(NsfA)
            self.col[R].remove(Nsf)

            if self.G is not None:
                self.G.add_edge(NsfA, R, type='no_cycle')
        del self.row[Nsf]

        # now process all of the no_cycle edges
        for C in self.col[Nsf]:
            edge = self.edges.pop((Nsf, C))
            self.edges[NsfB, C] = self.mulE(CLG, edge)
            self.oplistE.append(
                OpComp(
                    "E_mul2",
                    tupleize.EdgeTuple(NsfB, C),
                    (
                        tupleize.EdgeTuple(NsfB, NsfA),
                        tupleize.EdgeTuple(Nsf, C),
                    ),
                )
            )
            self.oplistE.append(OpComp("E_del", tupleize.EdgeTuple(Nsf, C), ()))
            self.row2[C].add(NsfB)
            self.col2[NsfB].add(C)
            self.row[C].remove(Nsf)

            if self.G is not None:
                self.G.add_edge(C, NsfB, type='no_cycle')

        del self.col[Nsf]

        for R in self.row2[Nsf]:
            edge = self.edges.pop((R, Nsf))
            self.edges[R, NsfB] = edge
            self.oplistE.append(
                OpComp(
                    "E_assign",
                    tupleize.EdgeTuple(R, NsfB),
                    (tupleize.EdgeTuple(R, Nsf),),
                )
            )
            self.oplistE.append(OpComp("E_del", tupleize.EdgeTuple(R, Nsf), ()))
            self.row2[NsfB].add(R)
            self.col2[R].add(NsfB)
            self.col2[R].remove(Nsf)
            if self.G is not None:
                self.G.add_edge(NsfB, R)
        del self.row2[Nsf]

        for C in self.col2[Nsf]:
            edge = self.edges.pop((Nsf, C))
            self.edges[NsfA, C] = edge
            self.oplistE.append(
                OpComp(
                    "E_assign",
                    tupleize.EdgeTuple(NsfA, C),
                    (tupleize.EdgeTuple(Nsf, C),),
                )
            )
            self.oplistE.append(OpComp("E_del", tupleize.EdgeTuple(Nsf, C), ()))
            self.row2[C].add(NsfA)
            self.col2[NsfA].add(C)
            self.row2[C].remove(Nsf)

            if self.G is not None:
                self.G.add_edge(C, NsfA)
        del self.col2[Nsf]

        self.nodes.remove(Nsf)

        if self.G is not None:
            self.G.remove_node(Nsf)
        return True

    def subinverse(self, R, C):
        """
        This computes the matrix element for an inverse from C to R
        """
        R = tupleize.tupleize(R)
        C = tupleize.tupleize(C)
        ops = []
        done = {}

        # since the graph is manefestly a DAG without cycles, can use depth first search
        # as a topological sort
        def recurse(node):
            cS = self.col2[node]
            # print('node', node, ' cS', cS)
            for cN in cS:
                if cN == C:
                    done[cN] = True
                elif cN not in done:
                    done[cN] = recurse(cN)

            used = False
            for cN in cS:
                if cN == C:
                    # load an edge into a node
                    ops.append(OpComp("N_edge", node, (tupleize.EdgeTuple(node, cN),)))
                    used = True
                elif done[cN]:
                    # load a node multiplied by an edge
                    # N_sum does not know if the target node has been loaded or exists already
                    # TODO, make this smarter by knowing if the target node has been loaded and
                    # using finer-grained code
                    ops.append(
                        OpComp("N_sum", node, (tupleize.EdgeTuple(node, cN), cN))
                    )
                    used = True
            return used

        recurse(R)
        ops.append(OpComp("N_ret", R, ()))
        return ops

    def inverse(self, R, Cin):
        """
        This computes the oplist to invert the matrix having loaded all or many of the columns C
        """
        R = tupleize.tupleize(R)
        Cin = {tupleize.tupleize(C) for C in Cin}
        ops = []
        done = {}

        # since the graph is manefestly a DAG without cycles, can use depth first search
        # as a topological sort
        def recurse(node):
            cS = self.col2[node]
            # print('node', node, ' cS', cS)
            for cN in cS:
                if cN in Cin:
                    done[cN] = True
                elif cN not in done:
                    done[cN] = recurse(cN)

            used = False
            for cN in cS:
                # load a node multiplied by an edge
                ops.append(
                    OpComp("N_sum", node, (tupleize.EdgeTuple(node, cN), cN))
                )
                used = True
            return used

        recurse(R)
        ops.append(OpComp("N_ret", R, ()))
        return ops


def purge_inplace(
    keep,
    row,
    col,
    edges,
):
    # can't actually purge, must color all nodes
    # from the exception set and then subtract the
    # remainder.
    # purging algorithms otherwise have to deal with
    # strongly connected components, which makes them
    # no better than coloring
    active_set = set()
    active_set_pending = set(keep)

    while active_set_pending:
        node = active_set_pending.pop()
        active_set.add(node)
        for snode in node:
            if snode not in active_set:
                active_set_pending.add(snode)

    full_set = set(row.keys()) | set(col.keys())
    purge = full_set - active_set
    purge_subgraph_inplace(row, col, edges, purge)


def purge_subgraph_inplace(
    purge,
    row,
    col,
    edges,
):
    for node in purge:
        for rN in row[node]:
            if rN not in purge:
                col[rN].remove(node)
        del row[node]

        for cN in col[node]:
            if cN not in purge and (cN, node):
                row[cN].remove(node)
                del edges[node, cN]
        del col[node]
    return
