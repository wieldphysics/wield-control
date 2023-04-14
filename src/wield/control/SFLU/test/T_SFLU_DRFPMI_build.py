#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
These tests demonstrate a heierachrical way to build connection graphs
for optical systems of mirrors and beamsplitters.
"""
import numpy as np
import networkx as nx
from wield.utilities.mpl import mplfigB

from wield.control.SFLU import nx2tikz
from wield.utilities.strings import padding_remove

from wield.control.SFLU import SFLU
from wield.control.SFLU import optics
from wield.control.SFLU.functions import neg

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


reduce_list = [
    'Xetm.fr.i',
    'Xetm.fr.o',
    'Xitm.bk.i',
    'Xitm.bk.o',
    'Yetm.fr.i',
    'Yetm.fr.o',
    'Yitm.bk.i',
    'Yitm.bk.o',

    'Xitm.fr.i',
    'Xitm.fr.o',
    'Yitm.fr.i',
    'Yitm.fr.o',

    'prm.fr.i',
    'prm.fr.o',
    'prm.bk.i',
    'prm.bk.o',

    'bs.frA.i',
    'bs.frA.o',
    'bs.frB.i',
    'bs.frB.o',
    'bs.bkA.i',
    'bs.bkA.o',
    'bs.bkB.i',
    'bs.bkB.o',

    'srm.fr.i',
    'srm.fr.o',
    'srm.bk.i',
    'srm.bk.o',
]



def T_SFLU_DRFPMI_build_show(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    ifo = optics.GraphElement()
    ifo.subgraph_add(
        'prm', optics.LossyBasisMirror(),
        translation_xy=(-25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'Xitm', optics.LossyBasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'Xetm', optics.LossyMirror(),
        translation_xy=(55, 0),
        rotation_deg=0
    )
    ifo.subgraph_add(
        'srm', optics.LossyBasisMirror(),
        translation_xy=(0, -25),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        'Yitm', optics.LossyBasisMirror(),
        translation_xy=(0, 25),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        'Yetm', optics.LossyMirror(),
        translation_xy=(0, 55),
        rotation_deg=90,
    )
    ifo.subgraph_add(
        'bs', optics.BeamSplitter(),
        translation_xy=(0, 0),
        rotation_deg=0
    )
    ifo.edges.update({
        ("prm.fr.i",  "bs.frA.o"): "prc.tau",
        ("bs.frA.i",  "prm.fr.o"): "prc.tau",

        ("srm.fr.i", "bs.bkB.o"): "sec.tau",
        ("bs.bkB.i", "srm.fr.o"): "sec.tau",

        ("Yitm.bk.i", "bs.frB.o"): 'BSY.tau',
        ("bs.frB.i",  "Yitm.bk.o"): 'BSY.tau',

        ("Xitm.bk.i", "bs.bkA.o"): 'BSX.tau',
        ("bs.bkA.i",  "Xitm.bk.o"): 'BSX.tau',

        ("Xetm.fr.i", "Xitm.fr.o"): "XARM.tau",
        ("Xitm.fr.i", "Xetm.fr.o"): "XARM.tau",

        ("Yetm.fr.i", "Yitm.fr.o"): "YARM.tau",
        ("Yitm.fr.i", "Yetm.fr.o"): "YARM.tau",
    })
    ifo['Yetm'].node_angle["fr.o"] = -45
    ifo['Yitm'].node_angle["bk.o"] = -45
    ifo['srm'].node_angle["bk.o"] = -45
    ifo['srm'].edges["bk.i", "bk.i.exc"] = "sec_to"
    ifo['srm'].edges["bk.o.tp", "bk.o"] = "sec_fr"
    ifo['srm'].edge_handedness["bk.o.tp", "bk.o"] = "r"

    ifo['srm'].locations["bk.i.exc"] = (15, -10)
    ifo['srm'].locations["bk.o.tp"] = (15, 10)

    ifo['prm'].edges["bk.i", "bk.i.exc"] = "1"
    ifo['prm'].edges["bk.o.tp", "bk.o"] = "1"
    ifo['prm'].node_angle["bk.i.exc"] = +45

    ifo['prm'].locations["bk.i.exc"] = (15, -10)
    ifo['prm'].locations["bk.o.tp"] = (15, 10)
    ifo['prm'].edges["fr.i.tp", "fr.i"] = "1"
    ifo['prm'].locations["fr.i.tp"] = (-5, 12)

    ifo['Xetm'].edges["fr.i.tp", "fr.i"] = "1"
    ifo['Xetm'].edges["fr.o", "fr.o.exc"] = "1"
    ifo['Xetm'].locations["fr.i.tp"] = (-5, 15)
    ifo['Xetm'].locations["fr.o.exc"] = (-5, -15)

    ifo['Yetm'].edges["fr.i.tp", "fr.i"] = "1"
    ifo['Yetm'].edge_handedness["fr.i.tp", "fr.i"] = "r"
    ifo['Yetm'].edges["fr.o", "fr.o.exc"] = "1"
    ifo['Yetm'].locations["fr.i.tp"] = (-5, 15)
    ifo['Yetm'].locations["fr.o.exc"] = (-5, -15)

    ifo['Xitm'].edges["bk.i.tp", "bk.i"] = "1"
    ifo['Xitm'].locations["bk.i.tp"] = (5, -12)
    ifo['Yitm'].edges["bk.i.tp", "bk.i"] = "1"
    ifo['Yitm'].locations["bk.i.tp"] = (5, -12)

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        derivatives=[
            'Xetm.fr.r',
            'Yetm.fr.r',
            'srm.fr.r',
            'BSX.tau',
            'BSY.tau',
        ],
        reduce_list=reduce_list,
        graph=True,
    )
    # match=False allows a reduced input/output set
    # sflu.graph_nodes_pos(ifo.build_locations(), match=True)
    ifo.update_sflu(sflu)
    G1 = sflu.G.copy()

    if True:
        yamlstr = sflu.convert_self2yamlstr()
        # print(yamlstr)
        with open(tpath_join('DRFPMI.yaml'), 'w') as F:
            F.write(yamlstr)
        sflu = SFLU.SFLU.convert_yamlstr2self(yamlstr)
    #sflu.graph_nodes_pos(DRFPMI_locs, match=True)

    print('inputs: ', sflu.inputs)
    print('outputs: ', sflu.outputs)
    print('nodes: ', sflu.nodes)
    # dprint('edges: ', sflu.edges)

    print('nodes')
    #print(sflu.graph_nodes_repr())
    #G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=0, dY=-2)
    sflu.reduce(*reduce_list)
    assert(not sflu.nodes)
    sflu.graph_reduce_auto_pos_io(lX=-30, rX=+30, Y=10, dY=-2)
    G2 = sflu.G.copy()
    dprint(G1.edges)

    nx2tikz.dump_pdf(
        [
            G1,
            G2,
        ],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        # preamble = preamble,
        scale='10pt',
    )

