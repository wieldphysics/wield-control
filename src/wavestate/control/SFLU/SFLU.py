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

from collections import defaultdict, namedtuple
from ..statespace import str_tup_keys as stk 

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
        nodes_cf = set()

        # this second set are the row and col edge sets that are cycle-free
        col2row_cf = defaultdict(set)
        row2col_cf = defaultdict(set)

        edges2 = dict()
        edges_original = dict()

        for (R, C), E in edges.items():
            R = stk.key_map(R)
            C = stk.key_map(C)
            edges2[stk.key_edge(R, C)] = E
            edges_original[stk.key_edge(R, C)] = E
            col2row[C].add(R)
            row2col[R].add(C)
            nodes.add(R)
            nodes.add(C)

        if self.G is not None:
            def to_label(val):
                return '$' + sp.latex(sp.var(str(val))) + '$'
            for (R, C), E in edges2.items():
                self.G.add_edge(C, R, label=to_label(E))
            for n in nodes:
                self.G.nodes[n]['label'] = to_label(n)
        # print('col2row', col2row)
        # print('row2col', row2col)

        if inputs is None:
            inputs = set()
            for rN in nodes:
                cS = row2col.get(rN, None)
                if cS is None or len(cS) == 0:
                    inputs.add(rN)
                elif len(cS) == 1 and rN in cS:
                    inputs.add(rN)

        if outputs is None:
            outputs = set()
            for cN in nodes:
                rS = col2row.get(cN, None)
                if rS is None or len(rS) == 0:
                    outputs.add(cN)
                elif len(rS) == 1 and cN in rS:
                    outputs.add(cN)

        for iN in inputs:
            cS = row2col[iN]
            if len(cS) == 1:
                assert iN in cS
                row2col_cf[iN].update(cS)
            else:
                assert len(cS) == 0
            del row2col[iN]

            rS = col2row[iN]
            dS = rS  # rS.intersection(outputs)
            col2row_cf[iN].update(dS)
            for rN in dS:
                row2col[rN].remove(iN)
                row2col_cf[rN].add(iN)

            rS.difference_update(dS)
            nodes.remove(iN)
            # nodes_cf.add(iN)

        for oN in outputs:
            rS = col2row[oN]
            if len(rS) == 1:
                assert oN in rS
                row2col_cf[oN].update(rS)
            else:
                assert len(rS) == 0
            del col2row[oN]

            cS = row2col[oN]
            dS = cS  # cS.intersection(inputs)
            row2col_cf[oN].update(dS)
            for cN in dS:
                col2row[cN].remove(oN)
                col2row_cf[cN].add(oN)
            cS.difference_update(dS)
            nodes.remove(oN)
            # nodes_cf.add(oN)

        self.col2row = col2row
        self.row2col = row2col

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
        self.nodes_cf = nodes_cf

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

    def graph_nodes_pos(self, pos, *nodes):
        if nodes:
            # assign pos to each node in nodes
            for n in nodes:
                n = stk.key_map(n)
                self.G.nodes[n]['pos'] = pos
        else:
            # assumes it is a dictionary
            for n, p in pos.items():
                n = stk.key_map(n)
                self.G.nodes[n]['pos'] = p
        return

    def graph_nodes_posX(self, posX, *nodes):
        for n in nodes:
            n = stk.key_map(n)
            pos = self.G.nodes[n].get('pos', (None, None))
            self.G.nodes[n]['pos'] = (posX, pos[1])

    def graph_nodes_posY(self, posY, *nodes):
        for n in nodes:
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
            n2 = stk.key_map(n)
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
        if self.row2col[Nsf] or self.col2row[Nsf]:
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
        self.nodes_cf.add(NsfB)
        self.nodes_cf.add(NsfA)

        if self.G is not None:
            self.G.remove_node(Nsf)
        return True


def purge_inplace(
    keep,
    col2row,
    row2col,
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

    full_set = set(col2row.keys()) | set(row2col.keys())
    purge = full_set - active_set
    purge_subgraph_inplace(col2row, row2col, edges, purge)


def purge_subgraph_inplace(
    purge,
    col2row,
    row2col,
    edges,
):
    for node in purge:
        for rN in col2row[node]:
            if rN not in purge:
                row2col[rN].remove(node)
        del col2row[node]

        for cN in row2col[node]:
            if cN not in purge and (cN, node):
                col2row[cN].remove(node)
                del edges[node, cN]
        del row2col[node]
    return


class SFLUCompute:
    def __init__(
        self,
        oplistE,
        edges,
        row2col,
        col2row,
        nodesizes={},
        defaultsize=None,
    ):
        self.oplistE     = oplistE
        self.edges       = edges
        self.row2col     = row2col
        self.col2row     = col2row
        self.defaultsize = defaultsize
        self.nodesizes   = nodesizes
        return

    def compute(self, **kwargs):
        Espace = dict()

        # load all of the initial values
        # TODO, allow this to include operations
        for ek, ev in self.edges.items():
            r, c = ek
            # r = tupleize.tupleize(r)
            # c = tupleize.tupleize(c)
            Espace[stk.key_edge(r, c)] = kwargs[ev]

        for op in self.oplistE:
            print(op)
            if op.op == "E_CLG":
                (arg,) = op.args
                E = Espace[arg]
                E.shape[-1]
                assert E.shape[-1] == E.shape[-2]
                I = np.eye(E.shape[-1])
                I = I.reshape((1,) * (len(E.shape) - 2) + (E.shape[-2:]))

                E2 = np.linalg.inv(I - E)

                Espace[op.targ] = E2

            elif op.op == "E_CLGd":
                (arg,) = op.args
                size = self.nodesizes.get(arg, self.defaultsize)
                I = np.eye(size)
                Espace[op.targ] = I

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
                Espace[op.targ] = E1 @ E2 @ E3 + EA

            elif op.op == "E_assign":
                (arg,) = op.args
                Espace[op.targ] = Espace[arg]

            elif op.op == "E_del":
                pass

            else:
                raise RuntimeError("Unrecognized Op {}".format(op))

        self.Espace = Espace


    def subinverse(self):
        return


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

    def subinverse_ops(self, R, C):
        """
        This computes the matrix element for an inverse from C to R

        TODO, the algorithm could/should use some work. Should make a copy of row2col_cf
        then deplete it
        """
        R = stk.key_map(R)
        C = stk.key_map(C)
        ops = []
        done = {}

        # since the graph is manefestly a DAG without cycles, can use depth first search
        # as a topological sort
        def recurse(node):
            cS = self.row2col_cf[node]
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
                    ops.append(OpComp("N_edge", node, (stk.key_edge(node, cN),)))
                    used = True
                elif done[cN]:
                    # load a node multiplied by an edge
                    # N_sum does not know if the target node has been loaded or exists already
                    # TODO, make this smarter by knowing if the target node has been loaded and
                    # using finer-grained code
                    ops.append(
                        OpComp("N_sum", node, (stk.key_edge(node, cN), cN))
                    )
                    used = True
            return used

        recurse(R)
        ops.append(OpComp("N_ret", R, ()))
        return ops

    def inverse_ops(self, R, Cin):
        """
        This computes the oplist to invert the matrix having loaded all or many of the columns C
        """
        R = stk.key_map(R)
        Cin = {stk.key_map(C) for C in Cin}
        ops = []
        done = {}

        # since the graph is manefestly a DAG without cycles, can use depth first search
        # as a topological sort
        def recurse(node):
            cS = self.row2col_cf[node]
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
                    OpComp("N_sum", node, (stk.key_edge(node, cN), cN))
                )
                used = True
            return used

        recurse(R)
        ops.append(OpComp("N_ret", R, ()))
        return ops
