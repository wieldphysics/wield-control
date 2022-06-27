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
import copy
from collections import defaultdict
from wavestate.bunch import Bunch

try:
    import sympy as sp
except ImportError:
    sp = None


class GraphElement(object):
    locations = {}
    edges = {}
    edge_handedness = {}
    node_angle = {}

    def __init__(self):
        # pull locations and edges out of the class
        self.locations = dict(self.locations)
        self.edges = dict(self.edges)
        self.edge_handedness = dict(self.edge_handedness)
        self.node_angle = dict(self.node_angle)

        # maps names to dictionaries containing the graph, transliation and rotation
        # and any other annotations
        self.subgraphs = {}

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, key):
        ks = key.split('.')
        obj = self.subgraphs[ks[0]].graph
        for k in ks[1:]:
            obj[k]
        return obj

    def subgraph_add(
        self,
        name,
        graph,
        translation_xy,
        rotation_deg=0
    ):
        self.subgraphs[name] = Bunch(
            graph=graph,
            translation_xy=translation_xy,
            rotation_deg=rotation_deg,
        )

    def build_locations(
        self
    ):
        return {k: v for k, v in self._build_locations(
            trans_xy=(0, 0),
            rot_deg=0,
        )}

    def _build_locations(self, trans_xy, rot_deg):
        for name, graphB in self.subgraphs.items():
            name = name + '.'
            for node, sub_xy in graphB.graph._build_locations(
                trans_xy=graphB.translation_xy,
                rot_deg=graphB.rotation_deg,
            ):
                yield (name + node), self.trans_rotate_xy(
                    trans_xy=trans_xy,
                    rot_deg=rot_deg,
                    sub_xy=sub_xy,
                )
        for name, sub_xy in self.locations.items():
            yield name, self.trans_rotate_xy(
                    trans_xy=trans_xy,
                    rot_deg=rot_deg,
                    sub_xy=sub_xy,
                )
        return

    def update_sflu(self, sflu):
        sflu.graph_nodes_pos(self.build_locations(), match=True)

        if sflu.G is not None:
            nodes, edges = self.build_properties()
            for node, ndict in nodes.items():
                for k, v in ndict.items():
                    sflu.G.nodes[node][k] = v
            for edge, edict in edges.items():
                for k, v in edict.items():
                    sflu.G.edges[edge][k] = v
        return

    def build_properties(
        self
    ):
        nodes, edges = self._build_properties(
            rot_deg=0,
        )
        return nodes, edges

    def properties(self, nodes: defaultdict, edges: defaultdict, rot_deg, **kw):
        """
        modify nodes and edges in place with any new changes
        """
        for node, angle in self.node_angle.items():
            # nodes is a  defaultdict(dict)
            nodes[node]['angle'] = angle

        for edge, handed in self.edge_handedness.items():
            edges[edge]['handed'] = handed
        return

    def _build_properties(self, rot_deg):
        nodes = defaultdict(dict)
        edges = defaultdict(dict)
        for name, graphB in self.subgraphs.items():
            name = name + '.'
            sub_nodes, sub_edges = graphB.graph._build_properties(
                rot_deg=(rot_deg + graphB.rotation_deg) % 360,
            )
            # TODO, make this a proper recursive update
            nodes.update({name+k: v for k, v in sub_nodes.items()})
            edges.update({(name+ck, name+rk): v for (rk, ck), v in sub_edges.items()})

        self.properties(
            nodes=nodes, edges=edges,
            rot_deg=rot_deg
        )
        return nodes, edges

    def trans_rotate_xy(self, trans_xy, rot_deg, sub_xy):
        x, y = sub_xy
        tx, ty = trans_xy
        phi = rot_deg/180 * np.pi
        c = np.cos(phi)
        s = np.sin(phi)
        return (tx + c*x - s*y), (ty + s*x + c*y)

    def build_edges(self):
        edges = dict()
        for name, graphB in self.subgraphs.items():
            name = name + '.'
            sub_edges = graphB.graph.build_edges()
            for (row, col), edge in sub_edges.items():
                if edge[0] == '.':
                    edges[name + row, name + col] = name + edge[1:]
                else:
                    edges[name + row, name + col] = edge

        edges.update(self.edges)
        return edges


class Mirror(GraphElement):

    locations = {
        "A.i": (-5, +5),
        "A.o": (-5, -5),
        "B.i": (+5, -5),
        "B.o": (+5, +5),
    }
    edges = {
        ("A.o", "B.i"): ".t",
        ("B.o", "A.i"): ".t",
        ("A.o", "A.i"): ".A.r",
        ("B.o", "B.i"): ".B.r",
    }

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            nodes["A.o"]['angle'] = +45
            nodes["B.i"]['angle'] = +45
            pass
        elif rot_deg < 135:
            edges[("A.o", "A.i")]['handed'] = 'r'
            edges[("B.o", "B.i")]['handed'] = 'r'
            nodes["A.i"]['angle'] = +45
            nodes["A.o"]['angle'] = +45
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            # edges[("A.o", "B.i")]['handed'] = 'r'
            # edges[("B.o", "A.i")]['handed'] = 'r'
            nodes["B.o"]['angle'] = +45
            nodes["A.i"]['angle'] = +45
            pass
        elif rot_deg < 270 + 45:
            edges[("A.o", "A.i")]['handed'] = 'r'
            edges[("B.o", "B.i")]['handed'] = 'r'
            nodes["B.i"]['angle'] = +45
            nodes["B.o"]['angle'] = +45
            # ~270deg
            pass
        else:
            pass
        super().properties(
            nodes=nodes,
            edges=edges,
            rot_deg=rot_deg,
            **kw
        )
        return


class BeamSplitter(GraphElement):

    locations = {
        "A1.i": (-10, +5),
        "A1.o": (-10, -5),
        "B1.i": (+10, -5),
        "B1.o": (+10, +5),
        "A2.o": (-5, +10),
        "A2.i": (+5, +10),
        "B2.i": (-5, -10),
        "B2.o": (+5, -10),
    }
    edges = {
        ("B1.o", "A1.i"): ".t",
        ("A1.o", "B1.i"): ".t",
        ("B2.o", "A2.i"): ".t",
        ("A2.o", "B2.i"): ".t",
        ("A2.o", "A1.i"): ".A.r",
        ("A1.o", "A2.i"): ".A.r",
        ("B1.o", "B2.i"): ".B.r",
        ("B2.o", "B1.i"): ".B.r",
    }



