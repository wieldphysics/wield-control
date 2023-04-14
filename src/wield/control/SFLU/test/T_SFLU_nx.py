#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import networkx as nx
from wield.utilities.mpl import mplfigB

from wield.control.SFLU import nx2tikz
from wield.utilities.strings import padding_remove

from wield.control.SFLU import SFLU

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


FP_edges = {
    ("a2_o", "a2_i"): "r_a2",  # make this a load operator
    ("a1_o", "a2_i"): "t_a",
    ("a1_o", "a1_i"): "r_a",
    ("a2_o", "a1_i"): "t_a",
    ("b1_o", "b1_i"): "r_b",
    ("b2_o", "b1_i"): "t_b",
    ("b2_o", "b2_i"): "r_b2",  # make this a load operator
    ("b1_o", "b2_i"): "t_b",
    ("b1_i", "a1_o"): "Lmat",
    ("a1_i", "b1_o"): "Lmat",
}


def T_networkx_basic(dprint, tpath_join, fpath_join):
    """
    Most basic checks about how networkx is used
    """
    G = nx.DiGraph()
    G.add_edges_from([(k1.replace('_', ''), k2.replace('_', '')) for k1, k2 in FP_edges.keys()])
    print(G.nodes())
    axB = mplfigB()
    pos = nx.nx_pydot.graphviz_layout(G, prog = 'dot')
    nx.set_node_attributes(G, pos, 'pos')
    print(G.nodes(data=True))
    nx.draw_networkx(G, arrows=True, pos = nx.get_node_attributes(G, 'pos'), ax = axB.ax0)
    axB.save(tpath_join('test'))

    nx2tikz.dump_tex(
        G,
        fname = tpath_join('test.tex'),
    )
    pass


edges_FP = {
    ("a2_o", "a2_i"): "r_a2",  # make this a load operator
    ("a1_o", "a2_i"): "t_a",
    ("a1_o", "a1_i"): "r_a",
    ("a2_o", "a1_i"): "t_a",
    ("b1_o", "b1_i"): "r_b",
    ("b2_o", "b1_i"): "t_b",
    ("b2_o", "b2_i"): "r_b2",  # make this a load operator
    ("b1_o", "b2_i"): "t_b",
    ("b1_i", "a1_o"): "l",
    ("a1_i", "b1_o"): "l",
}

def T_networkx_SFLU_FP(dprint, tpath_join, fpath_join):
    """
    most basic checks about how networkx is used to great tikz output.

    Reductions aren't performed much.
    """
    sflu = SFLU.SFLU(edges_FP, graph=True)

    print('nodes')
    print(sflu.graph_nodes_repr())

    sflu.graph_nodes_posY(-50, *[
        'b1_o',
        'b2_i',
        'a2_o',
        'a2_i',
        'b1_i',
        'b2_o',
        'a1_o',
        'a1_i'
    ])
    sflu.graph_nodes_posY(50, *[
        'a2_i',
        'b1_i',
        'b2_o',
        'a1_o',
    ])
    sflu.graph_nodes_posX(0, *['a2_o', 'a2_i'])
    sflu.graph_nodes_posX(50, *['a1_o', 'a1_i'])
    sflu.graph_nodes_posX(150, *['b1_o', 'b1_i'])
    sflu.graph_nodes_posX(200, *['b2_i', 'b2_o'])
    dprint(nx.get_node_attributes(sflu.G, 'pos'))

    G = sflu.G.copy()

    nx2tikz.dump_pdf(
        G,
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        preamble = preamble,
        scale='1pt',
    )
    pass

def T_networkx_SFLU_FP2(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    sflu = SFLU.SFLU(edges_FP, graph=True)
    sflu.graph_nodes_pos({
        'a2_i': (0, 50),
        'a2_o': (0, -50),
        'a1_o': (50, 50),
        'a1_i': (50, -50),
        'b1_i': (150, 50),
        'b1_o': (150, -50),
        'b2_o': (200, 50),
        'b2_i': (200, -50)
    })

    print('nodes')
    print(sflu.graph_nodes_repr())
    G1 = sflu.G.copy()
    sflu.reduce("a1_o")
    sflu.graph_nodes_pos({
        'L.a1_o': (-50, -100),
        # 'U.a1_o': (250, -100),
    })
    print(sflu.graph_nodes_repr())
    G2 = sflu.G.copy()
    sflu.reduce("b1_i")
    sflu.graph_nodes_pos({
        'L.b1_i': (-50, -120),
        'U.b1_i': (250, -120),
    })
    G3 = sflu.G.copy()
    sflu.reduce("a1_i")
    sflu.graph_nodes_pos({
        # 'L.a1_i': (-50, -140),
        'U.a1_i': (250, -140),
    })
    G4 = sflu.G.copy()
    sflu.reduce("b1_o")
    sflu.graph_nodes_pos({
        'L.b1_o': (-50, -160),
        'U.b1_o': (250, -160),
    })
    G5 = sflu.G.copy()
    nx2tikz.dump_pdf(
        [G1, G2, G3, G4, G5],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        preamble = preamble,
        scale='1pt',
    )

    import yaml
    print("DONE? ", sflu.nodes)
    oplist = []
    for op in sflu.oplistE:
        n, targ, args = op
        args2 = []
        for a in args:
            if isinstance(a, tuple):
                a = tuple(a)
            args2.append(a)
        odict = dict(op=n, targ=tuple(targ))
        if args2:
            odict['args'] = args2
        oplist.append(odict)

    print(yaml.safe_dump([oplist], default_flow_style=None))
    pass


locs = {
    "Pi": (-5, -5),
    "Po": (-5, +5),

    "Xi": (+25, +5),
    "Xo": (+25, -5),
    "Xi2": (+30, -5),
    "Xo2": (+30, +5),

    "eXi": (+40, +5),
    "eXo": (+40, -5),
    "eXo2": (+45, +5),
    "eXi2": (+45, -5),

    "Yi": (+5, +15),
    "Yo": (+15, +15),
    "Yo2": (+5, +20),
    "Yi2": (+15, +20),

    "eYi": (+5, +30),
    "eYo": (+15, +30),
    "eYo2": (+5, +35),
    "eYi2": (+15, +35),

    "Si": (+15, -15),
    "So": (+5, -15),
    "So2": (+15, -20),
    "Si2": (+5, -20),

    "BsPi": (-0, +5),
    "BsPo": (-0, -5),

    "BsXi": (+20, -5),
    "BsXo": (+20, +5),

    "BsYo": (+5, +10),
    "BsYi": (+15, +10),

    "BsSi": (+5, -10),
    "BsSo": (+15, -10),

    "SmE": (+5, -25),
    "SrO": (+15, -25),
    "XmE": (+40, -10),
    "YmE": (+20, +30),
}


edges = {
    ("BsPi", "BsXo") : r" edge[sflow=.4] node {$\tft_\bs$} ",
    ("BsXi", "BsPo") : r" edge[sflow=.4] node {$\tft_\bs$} ",

    ("BsYi", "BsSo") : r" edge[sflow=.6] node {$\tft_\bs$} ",
    ("BsSi", "BsYo") : r" edge[sflow=.6] node {$\tft_\bs$} ",

    ("BsPi", "BsYo") : r" edge[sflow=.5] node {$\tfr_\bs$} ",
    ("BsYi", "BsPo") : r" edge[sflow=.5] node {$\tfr_\bs$} ",

    ("BsSi", "BsXo") : r" edge[sflow=.5] node {$\tfr_\bs$} ",
    ("BsXi", "BsSo") : r" edge[sflow=.5] node {$\tfr_\bs$} ",

    ("BsPo", "Pi") : r" edge[sflow=.5] node {$\Tmat{L}_\prm$} ",
    ("Po", "BsPi") : r" edge[sflow=.5] node {$\Tmat{L}_\prm$} ",
    ("Pi", "Po") : r" edge[sflow=.5] node {$r_\prm$} ",

    ("BsYo", "Yi") : r" edge[sflow=.5] node {$\Tmat{U}_\xarm\Tmat{L}_\yarm$} ",
    ("Yo", "BsYi") : r" edge[sflow=.5] node {$\Tmat{L}_\yarm\Tmat{U}^{-1}_\yarm$} ",
    ("Yi", "Yo") : r" edge[sflow'=.5] node {$r_{iy}$} ",
    ("Yi2", "Yo2") : r" edge[sflow'=.5] node {$-r_{iy}$} ",
    ("Yi", "Yo2") : r" edge[sflow=.5] node {$t_{iy}$} ",
    ("Yi2", "Yo") : r" edge[sflow=.5] node {$t_{iy}$} ",

    ("eYi", "eYo") : r" edge[sflow'=.5] node {$-r_{ey}$} ",
    ("eYi2", "eYo2") : r" edge[sflow'=.5] node {$r_{ey}$} ",
    ("eYi", "eYo2") : r" edge[sflow=.5] node {$t_{ey}$} ",
    ("eYi2", "eYo") : r" edge[sflow=.5] node {$t_{ey}$} ",

    ("BsXo", "Xi") : r" edge[sflow=.5] node {$\Tmat{U}_\xarm\Tmat{L}_\xarm$} ",
    ("Xo", "BsXi") : r" edge[sflow=.5] node {$\Tmat{L}_\xarm\Tmat{U}^{-1}_\xarm$} ",
    ("Xi", "Xo") : r" edge[sflow'=.5] node {$r_{ix}$} ",
    ("Xi2", "Xo2") : r" edge[sflow'=.5] node {$-r_{ix}$} ",
    ("Xi", "Xo2") : r" edge[sflow=.5] node {$t_{ix}$} ",
    ("Xi2", "Xo") : r" edge[sflow=.5] node {$t_{ix}$} ",

    ("SmE", "Si2") : r" edge[sflow=.5] node {$\Tmat{1}$} ",
    ("So2", "SrO") : r" edge[sflow=.5] node {$\Tmat{1}$} ",
    ("XmE", "eXo") : r" edge[sflow=.5] node {$\Tmat{1}$} ",
    ("YmE", "eYo") : r" edge[sflow=.5] node {$\Tmat{1}$} ",

    ("Xo2", "eXi") : r" edge[sflow'=.5] node {$\Tmat{L}_a$} ",
    ("eXo", "Xi2") : r" edge[sflow'=.5] node {$\Tmat{L}_a$} ",

    ("Yo2", "eYi") : r" edge[sflow'=.5] node {$\Tmat{L}_a$} ",
    ("eYo", "Yi2") : r" edge[sflow'=.5] node {$\Tmat{L}_a$} ",

    ("eXi", "eXo") : r" edge[sflow'=.5] node {$-r_{ex}$} ",
    ("eXi2", "eXo2") : r" edge[sflow'=.5] node {$r_{ex}$} ",
    ("eXi", "eXo2") : r" edge[sflow=.5] node {$t_{ex}$} ",
    ("eXi2", "eXo") : r" edge[sflow=.5] node {$t_{ex}$} ",

    ("BsSo", "Si") : r" edge[sflow=.5] node {$\Tmat{L}_\src$} ",
    ("So", "BsSi") : r" edge[sflow=.5] node {$\Tmat{L}_\src$} ",
    ("Si", "So") : r" edge[sflow=.5] node {$r_\srm$} ",
    ("Si2", "So2") : r" edge[sflow=.5] node {$-r_\srm$} ",
    ("Si", "So2") : r" edge[sflow=.5] node {$t_\srm$} ",
    ("Si2", "So") : r" edge[sflow=.5] node {$t_\srm$} ",
}


def T_networkx_lg(dprint, tpath_join, fpath_join):
    """
    """
    G = nx.DiGraph()
    G.add_edges_from(edges.keys())
    print(G.nodes())
    axB = mplfigB()
    #pos = nx.nx_pydot.graphviz_layout(G, prog = 'dot')
    nx.set_node_attributes(G, locs, 'pos')
    nx.set_node_attributes(
        G,
        {},
        'pin'
    )
    nx.set_node_attributes(
        G,
        {
            "SmE": "$S$",
            "SrO": "$R$" ,
            "XmE": "$X$" ,
            "YmE": "$Y$" ,
        }, 'label'
    )
    print(G.nodes(data=True))
    nx.draw_networkx(G, arrows=True, pos = nx.get_node_attributes(G, 'pos'), ax = axB.ax0)
    nx.set_edge_attributes(G, edges, 'edge_text')
    axB.save(tpath_join('test'))

    nx2tikz.dump_tex(
        [G, G,],
        fname = tpath_join('test.tex'),
        preamble = preamble,
    )
    nx2tikz.dump_pdf(
        [G, G,],
        fname = tpath_join('testG.pdf'),
        preamble = preamble,
    )
    pass


preamble = padding_remove(r"""
\newcommand{\rmd}{\mathrm{d}}
\newcommand{\rme}{\mathrm{e}}
\newcommand{\rmi}{\mathrm{i}}
\newcommand{\hrms}{\ensuremath{h_{\text{rms}}}}
\newcommand{\htot}{\ensuremath{h_{\text{T}}}}

\newcommand{\FBW}{\ensuremath{f_{\text{bw}}}}
\newcommand{\rate}{\ensuremath{r}}
\newcommand{\msol}{\ensuremath{M_\odot}}
\newcommand{\omegaL}{\ensuremath{\nu}}

\newcommand{\E}[1]{
  \ensuremath{\mathrm{E}\!\left\{ #1 \right\}}
  }
\newcommand{\VAR}[1]{
  \ensuremath{\mathrm{VAR}\!\left\{ #1 \right\}}
  }
\newcommand{\COV}[1]{
  \ensuremath{\mathrm{COV}\!\left\{ #1 \right\}}
  }
\newcommand{\Tr}[1]{
  \ensuremath{\text{Tr}\left\{#1\right\}}
}

\newcommand{\stat}[1]{\ensuremath{\hat{#1}}}
\newcommand{\qhat}[1]{\ensuremath{\hat{#1}}}
\newcommand{\sqzOP}{\ensuremath{\qhat{\mathcal{S}}}}
\newcommand{\dispOP}{\ensuremath{\qhat{\mathcal{D}}}}
\newcommand{\bsOP}{\ensuremath{\qhat{\mathcal{B}}}}

\newcommand{\Marrow}[1]{\overset{\text{\tiny$\bm\leftrightarrow$}}{#1}}
\newcommand{\mat}[1]{\ensuremath{\mathbf{#1}}}
\newcommand{\Tmat}[1]{\ensuremath{\mathbbl{#1}}}
\newcommand{\Tvec}[1]{\overset{\raisebox{-2pt}{\text{\tiny$\bm\rightarrow$}}}{\mathbbl{#1}}}
\newcommand{\Marrowbbl}[1]{\overset{\raisebox{-1pt}{\text{\tiny$\bm\Leftrightarrow$}}}{#1}}
\newcommand{\Dmat}[1]{\ensuremath{\Marrowbbl{\mathbf{#1}}}}
\newcommand{\Mrarrowbbl}[1]{\overset{\raisebox{-1pt}{\text{\tiny$\bm\Rightarrow$}}}{#1}}
\newcommand{\Dvec}[1]{\ensuremath{\Mrarrowbbl{\mathbf{#1}}}}
\newcommand{\delay}{\ensuremath{L}}
\newcommand{\K}{\ensuremath{\mathcal{K}}}

\newcommand{\Psqz}{\ensuremath{\vec{E}_{\text{SQZ}}}}
%\newcommand{\Pread}{\ensuremath{\vec{E}_{\text{OMC}}}}
\newcommand{\Pread}{\ensuremath{\vec{\mathcal{P}_{2}}}}

\newcommand{\conj}[1]{\ensuremath{{#1}^*}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\MM}{\ensuremath{\Upsilon}}
\newcommand{\MMR}{\ensuremath{\Upsilon_{\text{R}}}}
\newcommand{\MMI}{\ensuremath{\Upsilon_{\text{I}}}}
\newcommand{\MMO}{\ensuremath{\Upsilon_{\text{O}}}}
\newcommand{\MMF}{\ensuremath{\Upsilon_{\text{F}}}}
\newcommand{\cMM}{\ensuremath{1{-}\MM}}
\newcommand{\cMMR}{\ensuremath{1{-}\MMR}}
\newcommand{\cMMI}{\ensuremath{1{-}\MMI}}
\newcommand{\cMMO}{\ensuremath{1{-}\MMO}}
\newcommand{\cMMF}{\ensuremath{1{-}\MMF}}

\newcommand{\LT}{\ensuremath{\Lambda_{\text{T}}}}
\newcommand{\ET}{\ensuremath{\eta_{\text{T}}}}
\newcommand{\LI}{\ensuremath{\Lambda_{\text{I}}}}
\newcommand{\LO}{\ensuremath{\Lambda_{\text{O}}}}

\newcommand{\sqzang}{\ensuremath{\phi}}
\newcommand{\sqzrot}{\ensuremath{\theta}}

\newcommand{\aFactor}{\ensuremath{\alpha}}
\newcommand{\bFactor}{\ensuremath{\beta}}
\newcommand{\psiR}{\ensuremath{\psi_\text{R}}}
\newcommand{\psiG}{\ensuremath{\psi_\text{G}}}
\newcommand{\psiS}{\ensuremath{\psi_\text{S}}}
\newcommand{\psiI}{\ensuremath{\psi_\text{I}}}
\newcommand{\psiO}{\ensuremath{\psi_\text{O}}}

%\newcommand{\Ophi}{\ensuremath{\overline{\phi}}}
\newcommand{\Ophi}{\ensuremath{\phi}}
\newcommand{\phiRMS}{\ensuremath{\phi_{\text{rms}}}}
\newcommand{\phiRMSsq}{\ensuremath{\phi^2_{\text{rms}}}}

\newcommand{\thetaRMS}{\ensuremath{\theta_{\text{rms}}}}
\newcommand{\thetaRMSsq}{\ensuremath{\theta^2_{\text{rms}}}}

\newcommand{\TLO}{\ensuremath{\Tvec{v}^{\dagger}}}
\newcommand{\TLOa}{\ensuremath{\Tvec{v}}}
\newcommand{\DLO}{\ensuremath{\Dvec{v}^\dagger}}
\newcommand{\DLOa}{\ensuremath{\Dvec{v}}}

\newcommand{\Hloss}{\ensuremath{\Tmat{H}_{\text{loss}}}}

\newcommand{\Mq}{\ensuremath{m_q}}
\newcommand{\Mp}{\ensuremath{m_p}}
\newcommand{\cMp}{\ensuremath{\conj{m}_p}}

\newcommand{\sql}{\ensuremath{\text{sql}}}
\newcommand{\subI}{\ensuremath{\text{I}}}
\newcommand{\subO}{\ensuremath{\text{O}}}
\newcommand{\subR}{\ensuremath{\text{R}}}
\newcommand{\subFC}{\ensuremath{\text{FC}}}
\newcommand{\subfc}{\ensuremath{\text{fc}}}
\newcommand{\IRO}{\ensuremath{\text{IRO}}}

\newcommand{\subF}{\ensuremath{\text{F}}}

\newcommand{\arm}{\ensuremath{\text{a}}}
\newcommand{\Arm}{\ensuremath{\text{A}}}
\newcommand{\src}{\ensuremath{\text{s}}}
\newcommand{\Src}{\ensuremath{\text{S}}}
\newcommand{\itm}{\ensuremath{\text{a}}}
\newcommand{\etm}{\ensuremath{\text{e}}}
\newcommand{\srm}{\ensuremath{\text{s}}}
\newcommand{\xarm}{\ensuremath{\text{x}}}
\newcommand{\yarm}{\ensuremath{\text{y}}}

\newcommand{\prm}{\ensuremath{\text{p}}}
\newcommand{\bs}{\ensuremath{\text{b}}}

\newcommand{\tfh}{\ensuremath{\mathfrak{h}}}
\newcommand{\tfr}{\ensuremath{\mathfrak{r}}}
\newcommand{\tft}{\ensuremath{\mathfrak{t}}}
\newcommand{\tfrH}{\ensuremath{\mathfrak{r}_{\text{hom}}}}

\newcommand{\rsrm}{\ensuremath{\Tmat{r}_\srm}}
\newcommand{\rprm}{\ensuremath{\Tmat{r}_\prm}}
\newcommand{\rxarm}{\ensuremath{\Tmat{r}_\xarm}}
\newcommand{\ryarm}{\ensuremath{\Tmat{r}_\yarm}}

\newcommand{\smu}{\ensuremath{{\,\mu}}}
\newcommand{\snu}{\ensuremath{{\,\nu}}}
""")

