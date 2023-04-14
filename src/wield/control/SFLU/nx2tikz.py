#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Export networkx graphs into tikz using signal flow style.

TODO, need to include the style files into TEXINPUTS or this will fail
"""
import subprocess
import tempfile
import os
from networkx import is_directed

from wield.utilities.strings import padding_remove


def dumps_tikz(g, scale='0.5em'):
    """Return TikZ code as `str` for `networkx` graph `g`."""
    s = []
    s.append(padding_remove(r"""
\begin{{tikzpicture}}[
  signal flow,
  pin distance=1pt,
  label distance=-2pt,
  x={scale}, y={scale},
  baseline=(current  bounding  box.center),
]""").format(scale=scale))

    def fix(n):
        n = str(n)
        return "{" + n.replace('.', '/') + "}"

    for n, d in g.nodes(data=True):
        n = fix(n)
        # label
        label = d.get('label', None)
        if not label:
            label = d.get('label_default', None)
        angle = d.get('angle', '-45')
        xy = d['pos']
        if xy is None:
            continue
        X, Y = xy

        if label is not None:
            label = 'pin={{{ang}: {label}}}'.format(ang=angle, label=label)
        # geometry
        color = d.get('color', None)
        shape = d.get('shape', 'nodeS')
        # style
        style = r', '.join(filter(None, [shape, label]))
        s.append(r'\node[{style}] ({n}) at ({X}, {Y}) {{}};'.format(style=style, n=n, X=X, Y=Y))

    s.append('')
    s.append(r'\path')

    for u, v, d in g.edges(data=True):
        u2 = fix(u)
        v2 = fix(v)

        edge_text = d.get('edge_text', None)

        handed = d.get('handed', 'l')
        dist = d.get('dist', 0.4)

        label = d.get('label', None)
        if not label:
            label = d.get('label_default', '')

        color = d.get('color', '')
        bend = d.get('bend', 0)
        suppress = d.get('suppress', False)
        if suppress:
            continue

        if edge_text is None:
            if label:
                label = ' node {{{label}}}'.format(label=label)
            if handed == 'l':
                etype = "sflow={}".format(dist)
            elif handed == 'r':
                etype = "sflow'={}".format(dist)
            else:
                raise NotImplementedError("unknown handedness")
            if bend != 0:
                bend = 'bend right={}'.format(bend)
            else:
                bend = None

            if u == v:
                loop = g.nodes[u].get('loop', 70)
                loop_width = g.nodes[u].get('loop_width', 70)
                loop = 'min distance=5mm, in={i}, out={o}, looseness=25'.format(i=loop + loop_width/2, o=loop - loop_width/2)
                bend = None
            else:
                loop = None

            style = r', '.join(filter(None, [etype, bend, loop, color]))
            s.append(r'({u}) edge[{style}]{label} ({v})'.format(style=style, label=label, u=u2, v=v2))
        else:
            s.append("({u}) {etext} ({v})".format(u=u2, v=v2, etext=edge_text))
    s.append(';')
    s.append(r'\end{tikzpicture}')

    return '\n'.join(s)


def _document(Gs, preamble='', **kwargs):
    """ Return `str` that contains a preamble and tikzpicture.

    The first positional argument is a graph or a list of graphs to combine into a picture.

    """
    if not isinstance(Gs, (list, tuple)):
        Gs = [Gs]

    tikz = []
    for g in Gs:
        tikz_tex = padding_remove(r"""
            \begin{{page}}
            {tikz}
            \end{{page}}
        """).format(tikz=dumps_tikz(g, **kwargs))
        tikz.append(tikz_tex)

    header = (
        padding_remove(r"""
\documentclass[multi=page]{standalone}
\pdfminorversion=7

\usepackage[english]{babel}
\usepackage[bbgreekl]{mathbbol}
\usepackage{amsfonts}

\DeclareSymbolFontAlphabet{\mathbb}{AMSb}
\DeclareSymbolFontAlphabet{\mathbbl}{bbold}

\usepackage{enumitem}
\usepackage{braket}
%\usepackage{microtype}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{txfonts}
\usepackage{comment}
\usepackage{bm}
\usepackage{mathtools}

\usepackage{tikzsignalflow}
\usepackage{sflow}

%\standaloneenv{tikzpicture}
"""))

    return (padding_remove(r"""
{header}
{preamble}

\begin{{document}}
{tikz}
\end{{document}}
    """)).format(
        header=header,
        preamble=preamble,
        tikz='\n\n'.join(tikz)
    )


def dump_tikz(g, fname, **kwargs):
    """Write TikZ picture as TeX file.

    The first positional argument is a graph or a list of graphs to combine into a picture.

    """
    s = dumps_tikz(g, **kwargs)
    with open(fname, 'w') as f:
        f.write(s)
    return


def dump_tex(g, fname, **kwargs):
    """Write TeX document.

    The first positional argument is a graph or a list of graphs to combine into a picture.

    """
    s = _document(g, **kwargs)
    with open(fname, 'w') as f:
        f.write(s)
    return


def dump_pdf(g, fname, texname = None, **kwargs):
    """Write PDF from TeX document using pdflatex.

    The first positional argument is a graph or a list of graphs to combine into a picture.

    """
    fpath, fbase = os.path.split(fname)

    s = _document(g, **kwargs)
    # typeset
    with tempfile.TemporaryDirectory(dir=fpath) as dname:
        if texname is None:
            base, ext = os.path.splitext(fbase)
            texname = os.path.join(dname, base + '.tex')
            texbase, ext = os.path.splitext(texname)
        else:
            base, ext = os.path.splitext(os.path.split(texname)[1])
            texbase = os.path.join(dname, base)

        with open(texname, 'w') as f:
            f.write(s)

        dtikz = os.path.join(os.path.split(__file__)[0], 'tikz') 
        files = [
             'sflow.sty',
            'tikzlibrarysignalflowstyles.code.tex',
             'tikzsignalflow.sty'
        ]
        for f in files:
            os.link(os.path.join(dtikz, f),  os.path.join(dname, f))

        opt = ['pdflatex', '-interaction=batchmode', '-output-directory=' + dname, texname]
        try:
            subprocess.run(opt, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # dump the log if needed
            with open(texbase + '.log', 'r') as f:
                output = f.read()
                print(output)
        else:
            os.rename(texbase + '.pdf', fname)
    return

