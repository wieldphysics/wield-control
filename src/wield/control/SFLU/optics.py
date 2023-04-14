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
import copy
from collections import defaultdict
from wield.bunch import Bunch

try:
    import sympy as sp
except ImportError:
    sp = None


class GraphElement(object):

    def __init__(self, **kw):

        # pull locations and edges out of the class
        self.locations = dict()
        self.edges = dict()
        self.edge_handedness = dict()
        self.node_angle = dict()
        # self.node_properties = defaultdict(dict)

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

        # shouldn't be needed
        # if sflu.G is not None:
        #     nodes, edges = self.build_properties()
        #     for node, ndict in nodes.items():
        #         for k, v in ndict.items():
        #             sflu.G.nodes[node][k] = v
        #     for edge, edict in edges.items():
        #         for k, v in edict.items():
        #             sflu.G.edges[edge][k] = v
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

    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "fr.i": (-4, +5),
            "fr.o": (-4, -5),
            "bk.i": (+4, -5),
            "bk.o": (+4, +5),
        })
        self.edges.update({
            ("fr.o", "bk.i"): ".t",
            ("bk.o", "fr.i"): ".t",
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
        })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            nodes["fr.o"]['angle'] = +45
            nodes["bk.i"]['angle'] = +45
            pass
        elif rot_deg < 135:
            edges[("fr.o", "fr.i")]['handed'] = 'r'
            edges[("bk.o", "bk.i")]['handed'] = 'r'
            nodes["fr.i"]['angle'] = +45
            nodes["fr.o"]['angle'] = +45
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            # edges[("fr.o", "bk.i")]['handed'] = 'r'
            # edges[("bk.o", "fr.i")]['handed'] = 'r'
            nodes["bk.o"]['angle'] = +45
            nodes["fr.i"]['angle'] = +45
            pass
        elif rot_deg < 270 + 45:
            edges[("fr.o", "fr.i")]['handed'] = 'r'
            edges[("bk.o", "bk.i")]['handed'] = 'r'
            nodes["bk.i"]['angle'] = +45
            nodes["bk.o"]['angle'] = +45
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


class BasisMirror(Mirror):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.edges.update({
            ("bk.o", "fr.i"): ".fr.t",
            ("fr.o", "bk.i"): ".bk.t",
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
        })



class LossyMirror(Mirror):
    """
    Creates a mirror with extra nodes to add losses on reflection.

    The reflectivity or transmissivity must be reduced to incorporate the loss, So
    A.r**2 + t**2 + A.l**2 = 1
    B.r**2 + t**2 + B.l**2 = 1
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frL.i": (-2, -10),
            "bkL.i": (+2, +10),
        })
        self.edges.update({
            ("fr.o", "frL.i"): ".fr.l",
            ("bk.o", "bkL.i"): ".bk.l",
        })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            edges[("fr.o", "frL.i")]['handed'] = 'r'
            edges[("fr.o", "frL.i")]['dist'] = 0.2

            nodes["bkL.i"]['angle'] = +45
            pass
        elif rot_deg < 135:
            edges[("fr.o", "frL.i")]['dist'] = 0.2
            nodes["frL.i"]['angle'] = +45
            nodes["bkL.i"]['angle'] = -135
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            edges[("bk.o", "bkL.i")]['handed'] = 'r'
            edges[("bk.o", "bkL.i")]['dist'] = 0.2

            nodes["frL.i"]['angle'] = +45
            pass
        elif rot_deg < 270 + 45:
            nodes["frL.i"]['angle'] = -135
            nodes["bkL.i"]['angle'] = +45
            edges[("bk.o", "bkL.i")]['dist'] = 0.2
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


class LossyBasisMirror(LossyMirror, BasisMirror, Mirror):
    pass


class BeamSplitter(GraphElement):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frA.i": (-10, +5),
            "frA.o": (-10, -5),
            "bkA.i": (+10, -5),
            "bkA.o": (+10, +5),
            "frB.o": (-5, +10),
            "frB.i": (+5, +10),
            "bkB.i": (-5, -10),
            "bkB.o": (+5, -10),
        })
        self.edges.update({
            ("bkA.o", "frA.i"): ".t",
            ("frA.o", "bkA.i"): ".t",
            ("bkB.o", "frB.i"): ".t",
            ("frB.o", "bkB.i"): ".t",
            ("frB.o", "frA.i"): ".fr.r",
            ("frA.o", "frB.i"): ".fr.r",
            ("bkA.o", "bkB.i"): ".bk.r",
            ("bkB.o", "bkA.i"): ".bk.r",
        })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            pass
        elif rot_deg < 135:
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            pass
        elif rot_deg < 270 + 45:
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


class LossyBeamSplitter(BeamSplitter):
    """
    Creates a mirror with extra nodes to add losses on reflection.

    The reflectivity or transmissivity must be reduced to incorporate the loss, So
    A.r**2 + t**2 + A.l**2 = 1
    B.r**2 + t**2 + B.l**2 = 1
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frAL.i": (-2, -10),
            "frBL.i": (-2, -10),
            "bkAL.i": (+2, +10),
            "bkBL.i": (+2, +10),
        })
        self.edges.update({
            ("frA.o", "frAL.i"): ".fr.l",
            ("bkA.o", "bkAL.i"): ".bk.l",
            ("frB.o", "frBL.i"): ".fr.l",
            ("bkB.o", "bkBL.i"): ".bk.l",
        })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            pass
        elif rot_deg < 135:
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            pass
        elif rot_deg < 270 + 45:
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


class Reflection(GraphElement):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frA.i": (-10, +5),
            "frA.o": (-10, -5),
            "frB.i": (+5, +10),
            "frB.o": (-5, +10),
        })
        self.edges.update({
            ("frB.o", "frA.i"): ".fr.r",
            ("frA.o", "frB.i"): ".fr.r",
        })


class LossyReflection(GraphElement):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frA.i": (-10, +5),
            "frA.o": (-10, -5),
            "frAL.i": (-3, -2),
            "frB.i": (+5, +10),
            "frB.o": (-5, +10),
            "frBL.i": (-13, +12),
        })
        self.edges.update({
            ("frB.o", "frA.i"): ".fr.r",
            ("frA.o", "frB.i"): ".fr.r",
            ("frA.o", "frAL.i"): ".fr.l",
            ("frB.o", "frBL.i"): ".fr.l",
        })



