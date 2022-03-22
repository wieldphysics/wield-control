#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import networkx as nx
from wavestate.utilities.mpl import mplfigB

from wavestate.control.SFLU import nx2tikz
from wavestate.utilities.strings import padding_remove

from wavestate.control.SFLU import SFLU
from wavestate.control.SFLU.functions import neg

from wavestate.pytest.fixtures import (  # noqa: F401
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

    nx2tikz.dump_pdf(
        G1,
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        preamble = preamble,
        scale='10pt',
    )

def T_SFLU_DRFPMI_working(dprint, tpath_join, fpath_join):
    """
    Show a graph reduction using networkx+tikz
    """
    sflu = SFLU.SFLU(
        DRFPMI_edges,
        graph=True,
        inputs = [
            'srm.B.i.exc',
            'Y.etm.A.o.exc',
            'X.etm.A.o.exc',
            'prm.A.i.exc',
            'srm.B.i.exc',
        ],
        outputs = [
            'srm.B.o.tp',
            'prm.A.o.tp',
            'X.etm.A.o.tp',
            'Y.etm.A.o.tp',
            'srm.B.o.tp',
        ],
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

        'srm.A.i',
        'srm.A.o',
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

        'srm.B.i',
        'srm.B.o',
    ]

    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=0, dY=-2)
    sflu.reduce(*reduce_list)
    print('nodes: ', sflu.nodes)
    sflu.graph_reduce_auto_pos_io(lX=-30, rX=+30, Y=-5, dY=-5)

    for rN, cS in sflu.row2col_cf.items():
        for cN in cS:
            sflu.G.edges[cN, rN]['color'] = 'blue'
    for cN, rS in sflu.col2row_cf.items():
        for rN in rS:
            sflu.G.edges[cN, rN]['color'] = 'red'
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G2],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.tex'),
        preamble = preamble,
        scale='10pt',
    )

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
    return



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
