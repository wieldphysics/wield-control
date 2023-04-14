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
import networkx as nx
from wield.utilities.mpl import mplfigB

from wield.control.SFLU import nx2tikz
from wield.utilities.strings import padding_remove

from wield.control.SFLU import SFLU
from wield.control.SFLU.functions import neg

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


DRFPMI_locs = {
    "prm.A.i": (-10, +5),
    "prm.A.o": (-10, -5),
    "prm.B.i": (-5, -5),
    "prm.B.o": (-5, +5),
    "prm.A.i.exc": (-20, +5),
    "prm.A.o.tp": (-20, -5),

    "X.itm.A.i": (+25, +5),
    "X.itm.A.o": (+25, -5),
    "X.itm.B.i": (+30, -5),
    "X.itm.B.o": (+30, +5),

    "X.etm.A.i": (+40, +5),
    "X.etm.A.o": (+40, -5),
    "X.etm.B.o": (+45, +5),
    "X.etm.B.i": (+45, -5),

    "Y.itm.A.i": (+5, +15),
    "Y.itm.A.o": (+15, +15),
    "Y.itm.B.o": (+5, +20),
    "Y.itm.B.i": (+15, +20),

    "Y.etm.A.i": (+5, +30),
    "Y.etm.A.o": (+15, +30),
    "Y.etm.B.o": (+5, +35),
    "Y.etm.B.i": (+15, +35),

    "srm.A.i": (+15, -15),
    "srm.A.o": (+5, -15),
    "srm.B.o": (+15, -20),
    "srm.B.i": (+5, -20),

    "BS.A1.i": (-0, +5),
    "BS.A1.o": (-0, -5),

    "BS.B1.i": (+20, -5),
    "BS.B1.o": (+20, +5),

    "BS.A2.o": (+5, +10),
    "BS.A2.i": (+15, +10),

    "BS.B2.i": (+5, -10),
    "BS.B2.o": (+15, -10),

    "srm.B.i.exc": (+5, -25),
    "srm.B.o.tp": (+15, -25),
    "X.etm.A.o.exc": (+38, -10),
    "X.etm.A.o.tp": (+41, -8),
    "Y.etm.A.o.exc": (+20, +30),
    "Y.etm.A.o.tp": (+20, +32),
}

DRFPMI_edges = {
    ("BS.B1.o"      ,  "BS.A1.i"       ): "BS.t",
    ("BS.A1.o"      ,  "BS.B1.i"       ): "BS.t",
    ("BS.B2.o"      ,  "BS.A2.i"       ): "BS.t",
    ("BS.A2.o"      ,  "BS.B2.i"       ): "BS.t",
    ("BS.A2.o"      ,  "BS.A1.i"       ): "BS.r",
    ("BS.A1.o"      ,  "BS.A2.i"       ): "BS.r",
    ("BS.B1.o"      ,  "BS.B2.i"       ): "-BS.r",
    ("BS.B2.o"      ,  "BS.B1.i"       ): "-BS.r",

    ("prm.A.o"      ,  "prm.B.i"       ): "prm.t",
    ("prm.B.o"      ,  "prm.A.i"       ): "prm.t",
    ("prm.A.o"      ,  "prm.A.i"       ): "prm.r",
    ("prm.B.o"      ,  "prm.B.i"       ): "-prm.r",

    ("srm.B.o"      ,  "srm.A.i"       ): "srm.t",
    ("srm.A.o"      , "srm.B.i"        ): "srm.t",
    ("srm.A.o"      , "srm.A.i"        ): "srm.r",
    ("srm.B.o"      ,  "srm.B.i"       ): "-srm.r",

    ("Y.itm.A.o"    ,  "Y.itm.A.i"     ): "Y.itm.r",
    ("Y.itm.B.o"    ,  "Y.itm.B.i"     ): "-Y.itm.r",
    ("Y.itm.B.o"    ,  "Y.itm.A.i"     ): "Y.itm.t",
    ("Y.itm.A.o"    ,  "Y.itm.B.i"     ): "Y.itm.t",

    ("Y.etm.A.o"    ,  "Y.etm.A.i"     ): "Y.etm.r",
    ("Y.etm.B.o"    ,  "Y.etm.B.i"     ): "-Y.etm.r",
    ("Y.etm.B.o"    ,  "Y.etm.A.i"     ): "Y.etm.t",
    ("Y.etm.A.o"    ,  "Y.etm.B.i"     ): "Y.etm.t",

    ("X.itm.A.o"    ,  "X.itm.A.i"     ): "X.itm.r",
    ("X.itm.B.o"    ,  "X.itm.B.i"     ): "-X.itm.r",
    ("X.itm.B.o"    ,  "X.itm.A.i"     ): "X.itm.t",
    ("X.itm.A.o"    ,  "X.itm.B.i"     ): "X.itm.t",

    ("X.etm.A.o"    ,  "X.etm.A.i"     ): "X.etm.r",
    ("X.etm.B.o"    ,  "X.etm.B.i"     ): "-X.etm.r",
    ("X.etm.B.o"    ,  "X.etm.A.i"     ): "X.etm.t",
    ("X.etm.A.o"    ,  "X.etm.B.i"     ): "X.etm.t",

    ("prm.B.i"      ,  "BS.A1.o"       ): "prc.tau",
    ("BS.A1.i"      ,  "prm.B.o"       ): "prc.tau",

    ("srm.A.i"      , "BS.B2.o"        ): "src.tau",
    ("BS.B2.i"      ,  "srm.A.o"       ): "src.tau",
     # ("BS.B2.i"      , "BS.B2.o"        ): "src.tau",
     # ("srm.A.i"      , "srm.A.o"        ): "1",

    ("Y.itm.A.i"    ,  "BS.A2.o"       ): ('*', 'BS_Y.tau'),
    ("BS.A2.i"      ,  "Y.itm.A.o"     ): ('*', 'BS_Y.tau'),

    ("X.itm.A.i"    ,  "BS.B1.o"       ): ('*', 'BS_X.tau'),
    ("BS.B1.i"      ,  "X.itm.A.o"     ): ('*', 'BS_X.tau'),

    ("X.etm.A.i"    ,  "X.itm.B.o"     ): "XARM.tau",
    ("X.itm.B.i"    ,  "X.etm.A.o"     ): "XARM.tau",

    ("Y.etm.A.i"    ,  "Y.itm.B.o"     ): "YARM.tau",
    ("Y.itm.B.i"    ,  "Y.etm.A.o"     ): "YARM.tau",

    ("X.etm.A.o"    ,  "X.etm.A.o.exc" ): "1",
    ("Y.etm.A.o"    ,  "Y.etm.A.o.exc" ): "1",
    ("X.etm.A.o.tp" ,  "X.etm.A.o"     ): "1",
    ("Y.etm.A.o.tp" ,  "Y.etm.A.o"     ): "1",

    ("srm.B.i"      ,  "srm.B.i.exc"   ): "1",
    ("srm.B.o.tp"   ,  "srm.B.o"       ): "1",
    ("prm.A.i"      ,  "prm.A.i.exc"   ): "1",
    ("prm.A.o.tp"   ,  "prm.A.o"       ): "1",
}

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

    'BS.A1.i',
    'BS.A1.o',
    'BS.A2.i',
    'BS.A2.o',
    'BS.B1.i',
    'BS.B1.o',
    'BS.B2.i',
    'BS.B2.o',

    'srm.A.i',
    'srm.A.o',
    'srm.B.i',
    'srm.B.o',
]


in_out = dict(
    inputs = [
        'srm.B.i.exc',
        'Y.etm.A.o.exc',
        'X.etm.A.o.exc',
        'prm.A.i.exc',
        'srm.B.i.exc',
        'X.etm.B.i',
        'Y.etm.B.i',
    ],
    outputs = [
        'srm.B.o.tp',
        'prm.A.o.tp',
        'X.etm.A.o.tp',
        'Y.etm.A.o.tp',
        'srm.B.o.tp',
        'X.etm.B.o',
        'Y.etm.B.o',
    ],
)


def T_SFLU_DRFPMI_show_full(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    sflu = SFLU.SFLU(
        DRFPMI_edges,
        graph=True,
    )
    # match=False allows a reduced input/output set
    sflu.graph_nodes_pos(DRFPMI_locs, match=True)
    #sflu.graph_nodes_pos(DRFPMI_locs, match=True)
    print('inputs: ', sflu.inputs)
    print('outputs: ', sflu.outputs)
    print('nodes: ', sflu.nodes)

    #print('nodes')
    #print(sflu.graph_nodes_repr())
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=0, dY=-2)
    sflu.reduce(*reduce_list)
    sflu.graph_reduce_auto_pos_io(lX=-30, rX=+30, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        # preamble = preamble,
        scale='10pt',
    )

def T_SFLU_DRFPMI_show_sub(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    sflu = SFLU.SFLU(
        DRFPMI_edges,
        graph=True,
        **in_out,
    )
    # match=False allows a reduced input/output set
    sflu.graph_nodes_pos(DRFPMI_locs, match=True)
    G0 = sflu.G.copy()
    #sflu.graph_nodes_pos(DRFPMI_locs, match=True)
    print('inputs: ', sflu.inputs)
    print('outputs: ', sflu.outputs)
    print('nodes: ', sflu.nodes)

    #print('nodes')
    #print(sflu.graph_nodes_repr())
    G1 = sflu.G.copy()

    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=0, dY=-2)
    sflu.reduce(*reduce_list)
    print('nodes: ', sflu.nodes)

    sflu.graph_reduce_auto_pos_io(lX=-30, rX=+30, Y=-5, dY=-5)

    G2 = sflu.G.copy()
    G3 = sflu.G.copy()
    for rN, cS in sflu.row2col_cf.items():
        for cN in cS:
            G2.edges[cN, rN]['color'] = 'blue'
    for cN, rS in sflu.col2row_cf.items():
        for rN in rS:
            G2.edges[cN, rN]['color'] = 'red'

    if True:
        # this is to colorize the edges of G3 based on the computation
        comp = sflu.computer()

        edge_map = {}
        comp.edge_map(edge_map = edge_map, default = 1)
        print(edge_map)

        T_etm = 0
        T_itm = 0.0148
        T_prm = 0.03
        T_srm = 0.35
        emap = {
            '1': 1,
            'BS.r': 0.5**0.5,
            'BS.t': 0.5**0.5,
            'BS_X.tau': np.exp(np.pi*2j*0),
            'BS_Y.tau': np.exp(np.pi*2j*0),
            'X.etm.r': (1-T_etm)**0.5,
            'X.etm.t': T_etm**0.5,
            'X.itm.r': (1-T_itm)**0.5,
            'X.itm.t': T_itm**0.5,
            'XARM.tau':  np.exp(np.pi*2j*0),
            'Y.etm.r': (1-T_etm)**0.5,
            'Y.etm.t': T_etm**0.5,
            'Y.itm.r': (1-T_itm)**0.5,
            'Y.itm.t': T_itm**0.5,
            'YARM.tau': np.exp(np.pi*2j*0),
            'prc.tau': np.exp(np.pi*2j*0),
            'prm.r': (1-T_prm)**0.5,
            'prm.t': T_prm**0.5,
            'src.tau': np.exp(np.pi*2j*0),
            'srm.r': (1-T_srm)**0.5,
            'srm.t': T_srm**0.5,
        }
        assert(set(edge_map.keys()) == set(emap.keys()))

        comp.compute(edge_map=emap)
        for rN, cN in comp.Espace.keys():
            try:
                G3.edges[cN, rN]['color'] = 'red'
            except KeyError:
                pass
    else:
        G3 = G2

    nx2tikz.dump_pdf(
        [G1, G2, G3],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        # preamble = preamble,
        scale='10pt',
    )


def T_SFLU_DRFPMI_serialize(dprint, tpath_join, fpath_join):
    sflu = SFLU.SFLU(
        DRFPMI_edges,
        **in_out,
    )
    sflu.reduce(*reduce_list)
    comp = sflu.computer()

    oplistE_yamlstr = comp.convert_oplistE2yamlstr()
    print(oplistE_yamlstr)

    assert(comp.convert_yamlstr2oplistE(oplistE_yamlstr) == comp.oplistE)

    comp_yamlstr = comp.convert_self2yamlstr()
    print(comp_yamlstr)

    comp2 = SFLU.SFLUCompute.from_yaml(comp_yamlstr)
    assert(comp2.oplistE == comp.oplistE)
    assert(comp2.edges == comp.edges)
    assert(comp2.row2col == comp.row2col)
    assert(comp2.col2row == comp.col2row)

    with open(tpath_join('DRFPMI.yaml'), 'w') as F:
        F.write(comp_yamlstr)
    return

def T_SFLU_DRFPMI_working(dprint, tpath_join, fpath_join):
    sflu = SFLU.SFLU(
        DRFPMI_edges,
        **in_out,
    )
    sflu.reduce(*reduce_list)
    comp = sflu.computer()
    print(comp.edges)

    edge_map = {}
    comp.edge_map(edge_map = edge_map, default = 1)
    print(edge_map)

    T_etm = 80e-6
    T_itm = 0.0148
    T_prm = 0.03
    T_srm = 0.35
    emap = {
        '1': 1,
        'BS.r': 0.5**0.5,
        'BS.t': 0.5**0.5,
        'BS_X.tau': np.exp(np.pi*0.5j),
        'BS_Y.tau': np.exp(np.pi*2j*0),
        'X.etm.r': (1-T_etm)**0.5,
        'X.etm.t': T_etm**0.5,
        'X.itm.r': (1-T_itm)**0.5,
        'X.itm.t': T_itm**0.5,
        'XARM.tau':  np.exp(np.pi*2j*0),
        'Y.etm.r': (1-T_etm)**0.5,
        'Y.etm.t': T_etm**0.5,
        'Y.itm.r': (1-T_itm)**0.5,
        'Y.itm.t': T_itm**0.5,
        'YARM.tau': np.exp(np.pi*2j*0),
        'prc.tau': np.exp(np.pi*2j*0),
        'prm.r': (1-T_prm)**0.5,
        'prm.t': T_prm**0.5,
        'src.tau': np.exp(np.pi*2j*0),
        'srm.r': (1-T_srm)**0.5,
        'srm.t': T_srm**0.5,
    }
    assert(set(edge_map.keys()) == set(emap.keys()))

    comp.compute(edge_map=emap)
    dprint(list(comp.Espace.keys()))
    #results = comp.inverse_col(['srm.B.o.tp'], {'srm.B.i.exc':1})['srm.B.o.tp']
    results = comp.inverse_single('srm.B.o.tp', 'srm.B.i.exc')
    print("CALC: ", abs(results)**2)

    results = comp.inverse_row({'srm.B.o.tp': None}, {
        'srm.B.i.exc',
        'prm.A.i.exc',
        'X.etm.B.i',
        'Y.etm.B.i',
    })
    print("CALC: ", {k : abs(r)**2 for k, r in results.items()}, sum(abs(r)**2 for k, r in results.items()))
    pass



