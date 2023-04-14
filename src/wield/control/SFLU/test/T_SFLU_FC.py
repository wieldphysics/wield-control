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
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import sympy
from wield.utilities.mpl import mplfigB
from wield.utilities.np import logspaced

from wield.control.SFLU import SFLU

from quantum_lib import mats_planewave as ilib


from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
)


def T_SFLU_FP(dprint, tpath_join, fpath_join):
    """
    Setup a cavity with fields
    a2:a1 ---- b1:b2
    """

    edges = {
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
    sflu = SFLU.SFLU(edges)
    print("inputs", sflu.inputs)
    print("outputs", sflu.outputs)

    print("row1", sflu.row)
    print("row2", sflu.row2)
    print("col1", sflu.col)
    print("col2", sflu.col2)
    print("-----------------------")
    sflu.reduce("a1_i")
    sflu.reduce("b1_i")
    sflu.reduce("b1_o")
    sflu.reduce("a1_o")
    # sflu.reduce('a2_i')
    # sflu.reduce('a2_o')
    # sflu.reduce('b2_i')
    # sflu.reduce('b2_o')
    print("-----------------------")

    print("row1", sflu.row)
    print("row2", sflu.row2)
    print("col1", sflu.col)
    print("col2", sflu.col2)
    dprint(sflu.oplistE)
    oplistN = sflu.subinverse("a2_o", "a2_i")
    dprint(oplistN)

    F_Hz = logspaced(1, 1e5, 1000)
    L_m = 4000
    i2pi = 2j * np.pi
    R_a = 0.01
    c_m_s = 3e8

    Espace = computeLU(
        oplistE=sflu.oplistE,
        edges=sflu.edges_original,
        nodesizes={},
        defaultsize=2,
        **dict(
            r_a=ilib.Id * R_a ** 0.5,
            r_a2=ilib.Id * -(R_a ** 0.5),
            t_a=ilib.Id * (1 - R_a) ** 0.5,
            r_b=ilib.Id * 1,
            t_b=ilib.Id * 0,
            r_b2=ilib.Id * -1,
            # Lmat = ilib.Mrotation(2 * np.pi * F_Hz * c_m_s / L_m),
            Lmat=ilib.diag(np.exp(-i2pi * F_Hz * L_m / c_m_s)),
        ),
    )
    # print("ESPACE: ", Espace)
    # print(list(Espace.keys()))

    SI = compute_subinverse(
        oplistN=oplistN,
        Espace=Espace,
        nodesizes={},
        defaultsize=2,
    )
    axB = mplfigB(Nrows=2)
    axB.ax0.loglog(F_Hz, abs(SI[:, 0, 0]))
    axB.ax1.semilogx(F_Hz, np.angle(SI[:, 0, 0]))
    axB.save(tpath_join("reflSI"))
    # print(Espace)
    return


