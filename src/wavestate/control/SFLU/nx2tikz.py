#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3
# Originally from https://github.com/johnyf/nx2tikz
"""Export NetworkX graphs to TikZ graphs with automatic layout."""
import subprocess
import tempfile
import os
from networkx import is_directed

from wavestate.utilities.strings import padding_remove


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
        return "{" + n.replace('=', '').replace('.', '+') + "}"

    for n, d in g.nodes(data=True):
        n = fix(n)
        # label
        label = d.get('label', None)
        angle = d.get('angle', '-45')
        X, Y = d['pos']

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
        u = fix(u)
        v = fix(v)

        edge_text = d.get('edge_text', None)
        if edge_text is None:
            handed = d.get('handed', 'l')
            dist = d.get('handed', 0.5)

            label = d.get('label', '')
            color = d.get('color', '')
            if label:
                label = ' node {{{label}}}'.format(label=label)
            if handed == 'l':
                etype = "sflow={}".format(dist)
            elif handed == 'r':
                etype = "sflow'={}".format(dist)
            else:
                raise NotImplementedError("unknown handedness")

            style = r', '.join(filter(None, [etype, color]))
            s.append(r'({u}) edge[{style}]{label} ({v})'.format(style=style, label=label, u=u, v=v))
        else:
            s.append("({u}) {etext} ({v})".format(u=u, v=v, etext=edge_text))
    s.append(';')
    s.append(r'\end{tikzpicture}')

    return '\n'.join(s)


def _document(*Gs, preamble='', **kwargs):
    """Return `str` that contains a preamble and tikzpicture."""
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
    """Write TikZ picture as TeX file."""
    s = dumps_tikz(g, **kwargs)
    with open(fname, 'w') as f:
        f.write(s)
    return


def dump_tex(*g, fname, **kwargs):
    """Write TeX document (use this as an example)."""
    s = _document(*g, **kwargs)
    with open(fname, 'w') as f:
        f.write(s)
    return


def dump_pdf(*g, fname, texname = None, **kwargs):
    fpath, fbase = os.path.split(fname)

    s = _document(*g, **kwargs)
    # typeset
    with tempfile.TemporaryDirectory(dir=fpath) as dname:
        if texname is None:
            base, ext = os.path.splitext(fbase)
            texname = os.path.join(dname, base + '.tex')
            texbase, ext = os.path.splitext(texname)
        else:
            base, ext = os.path.splitext(os.path.split(texname)[1])
            texbase = os.path.join(dname, base)
            print('texbase', texbase)

        with open(texname, 'w') as f:
            f.write(s)

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

