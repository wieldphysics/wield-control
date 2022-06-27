#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
These tests demonstrate a heierachrical way to build connection graphs
for optical systems of mirrors and beamsplitters.
"""
import numpy as np
import networkx as nx
from wavestate.utilities.mpl import mplfigB

from wavestate.control.SFLU import nx2tikz
from wavestate.utilities.strings import padding_remove

from wavestate.control.SFLU import SFLU
from wavestate.control.SFLU import optics
from wavestate.control.SFLU.functions import neg

from wavestate.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


reduce_list = [
    'X.etm.A.i',
    'X.etm.A.o',
    'X.itm.B.i',
    'X.itm.B.o',
    'Y.etm.A.i',
    'Y.etm.A.o',
    'Y.itm.B.i',
    'Y.itm.B.o',

    'X.itm.A.i',
    'X.itm.A.o',
    'Y.itm.A.i',
    'Y.itm.A.o',

    'prm.A.i',
    'prm.A.o',
    'prm.B.i',
    'prm.B.o',

    'bs.A1.i',
    'bs.A1.o',
    'bs.A2.i',
    'bs.A2.o',
    'bs.B1.i',
    'bs.B1.o',
    'bs.B2.i',
    'bs.B2.o',

    'srm.A.i',
    'srm.A.o',
    'srm.B.i',
    'srm.B.o',
]



def T_SFLU_DRFPMI_build_show(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    ifo = optics.GraphElement()
    mirror = optics.Mirror()
    beamsplitter = optics.BeamSplitter()
    ifo.subgraph_add(
        'prm', mirror.copy(),
        translation_xy=(-30, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'Xitm', mirror.copy(),
        translation_xy=(30, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'Xetm', mirror.copy(),
        translation_xy=(70, 0),
        rotation_deg=0
    )
    ifo.subgraph_add(
        'srm', mirror.copy(),
        translation_xy=(0, -30),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        'Yitm', mirror.copy(),
        translation_xy=(0, 30),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        'Yetm', mirror.copy(),
        translation_xy=(0, 70),
        rotation_deg=90,
    )
    ifo.subgraph_add(
        'bs', beamsplitter.copy(),
        translation_xy=(0, 0),
        rotation_deg=0
    )
    ifo.edges.update({
        ("prm.A.i"      ,  "bs.A1.o"       ): "prc.tau",
        ("bs.A1.i"      ,  "prm.A.o"       ): "prc.tau",

        ("srm.A.i"      , "bs.B2.o"        ): "src.tau",
        ("bs.B2.i"      , "srm.A.o"       ): "src.tau",

        ("Yitm.B.i"    ,  "bs.A2.o"       ): 'BSY.tau',
        ("bs.A2.i"      ,  "Yitm.B.o"     ): 'BSY.tau',

        ("Xitm.B.i"    ,  "bs.B1.o"       ): 'BSX.tau',
        ("bs.B1.i"      ,  "Xitm.B.o"     ): 'BSX.tau',

        ("Xetm.A.i"    ,  "Xitm.A.o"     ): "XARM.tau",
        ("Xitm.A.i"    ,  "Xetm.A.o"     ): "XARM.tau",

        ("Yetm.A.i"    ,  "Yitm.A.o"     ): "YARM.tau",
        ("Yitm.A.i"    ,  "Yetm.A.o"     ): "YARM.tau",
    })
    ifo['srm'].edges["B.i", "B.i.exc"] = "1"
    ifo['srm'].edges["B.o.tp", "B.o"] = "1"
    ifo['srm'].edge_handedness["B.o.tp", "B.o"] = "r"

    ifo['srm'].locations["B.i.exc"] = (5, -15)
    ifo['srm'].locations["B.o.tp"] = (5, 15)

    ifo['prm'].edges["B.i", "B.i.exc"] = "1"
    ifo['prm'].edges["B.o.tp", "B.o"] = "1"
    ifo['prm'].locations["B.i.exc"] = (5, -15)
    ifo['prm'].locations["B.o.tp"] = (5, 15)

    ifo['Xetm'].edges["A.i.tp", "A.i"] = "1"
    ifo['Xetm'].edges["A.o", "A.o.exc"] = "1"
    ifo['Xetm'].locations["A.i.tp"] = (-5, 15)
    ifo['Xetm'].locations["A.o.exc"] = (-5, -15)

    ifo['Yetm'].edges["A.i.tp", "A.i"] = "1"
    ifo['Yetm'].edge_handedness["A.i.tp", "A.i"] = "r"
    ifo['Yetm'].edges["A.o", "A.o.exc"] = "1"
    ifo['Yetm'].locations["A.i.tp"] = (-5, 15)
    ifo['Yetm'].locations["A.o.exc"] = (-5, -15)

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        derivatives=[
            'Xetm.A.r',
            'Yetm.A.r',
            'BSX.tau',
            'BSY.tau',
        ],
        graph=True,
    )
    # match=False allows a reduced input/output set
    # sflu.graph_nodes_pos(ifo.build_locations(), match=True)
    ifo.update_sflu(sflu)
    #sflu.graph_nodes_pos(DRFPMI_locs, match=True)

    print('inputs: ', sflu.inputs)
    print('outputs: ', sflu.outputs)
    print('nodes: ', sflu.nodes)

    #print('nodes')
    #print(sflu.graph_nodes_repr())
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=0, dY=-2)
    #sflu.reduce(*reduce_list)
    #sflu.graph_reduce_auto_pos_io(lX=-30, rX=+30, Y=-5, dY=-5)
    #G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [
            G1,
            #G2,
        ],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        # preamble = preamble,
        scale='10pt',
    )

