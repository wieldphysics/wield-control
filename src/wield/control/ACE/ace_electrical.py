#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
State Space System
"""

import numpy as np
import copy
from wield import declarative
from collections import defaultdict, Mapping
import functools
from numbers import Number

from wield.control.algorithms.statespace import dense
from wield.control.algorithms.statespace.dense.zpk_algorithms import zpk_cascade
from wield.control.algorithms.statespace.dense.xfer_algorithms import ss2xfer

from . import ACE


def zpkACE(Zr=(), Zc=(), Pr=(), Pc=(), k=1):
    ABCDs = zpk_cascade(zr=Zr, zc=Zc, pr=Pr, pc=Pc, k=k)
    syslist = []
    for (A, B, C, D, E) in ABCDs:
        syslist.append(ACE.ACE.from_ABCD(A, B, C, D, E))

    ace = ACE.ACE()
    for idx, sys in enumerate(syslist):
        ace.insert(sys, cmn="sys{}".format(idx))
    for idx in range(len(syslist) - 1):
        ace.bind_equal(
            {"sys{}.O".format(idx), "sys{}.I".format(idx + 1)},
            constr="s{}{}".format(idx, idx + 1),
        )
    ace.io_add("I", {"sys0.I": None})
    ace.io_add("O", {"sys{}.O".format(idx): None})
    return ace


pi2 = np.pi * 2


def op_amp(Gbw=1e7):
    ace = ACE.ACE()
    ace.insert(
        zpkACE(
            Pr=[
                -1 * pi2,
            ],
            k=Gbw,
        ),
        cmn="Vgain",
    )
    ace.insert(
        zpkACE(
            Pr=[
                -1 * pi2,
            ],
            Zr=[
                -100 * pi2,
            ],
            k=4e-9 * 100,
        ),
        cmn="Vnoise",
    )
    ace.insert(
        zpkACE(
            Pr=[
                -1 * pi2,
            ],
            Zr=[
                -30 * pi2,
            ],
            k=10e-12 * 100,
        ),
        cmn="Inoise",
    )
    ace.states_augment(N=1, st="posI", io=True)
    ace.states_augment(N=1, st="posV", io=True)
    ace.states_augment(N=1, st="negI", io=True)
    ace.states_augment(N=1, st="negV", io=True)
    ace.states_augment(N=1, st="outI", io=True)
    ace.states_augment(N=1, st="outV", io=True)
    ace.bind_equal({"outV", "Vgain.O"}, constr="Vgain.O")
    ace.bind_sum({"posI"}, constr="posI")
    ace.bind_equal({"negI", "Inoise.O"}, constr="negI")
    ace.bind_sum({"posV": -1, "negV": 1, "Vnoise.O": -1, "Vgain.I": 1}, constr="amp")
    ace.io_input("Vnoise.I")
    ace.io_input("Inoise.I")
    ace.noise_add("opamp", {"Vnoise.I", "Inoise.I"})

    ace.port_add("inP", type="electrical", flow="posI", potential="posV")
    ace.port_add("inN", type="electrical", flow="negI", potential="negV")
    ace.port_add("out", type="electrical", flow="outI", potential="outV")
    return ace


def electrical1port():
    ace = ACE.ACE()
    ace.states_augment(N=1, st="aI", io=True)
    ace.states_augment(N=1, st="aV", io=True)

    ace.port_add("a", type="electrical", flow="aI", potential="aV")
    return ace


def short():
    ace = electrical1port()
    ace.bind_sum({"aV"}, constr="V")
    return ace


def open():
    ace = electrical1port()
    ace.bind_sum({"aI"}, constr="V")
    return ace


def electrical2port():
    ace = ACE.ACE()
    ace.states_augment(N=1, st="aI", io=True)
    ace.states_augment(N=1, st="bI", io=True)
    ace.states_augment(N=1, st="aV", io=True)
    ace.states_augment(N=1, st="bV", io=True)

    ace.port_add("a", type="electrical", flow="aI", potential="aV")
    ace.port_add("b", type="electrical", flow="bI", potential="bV")
    return ace


def voltage_source1():
    ace = electrical1port()
    ace.states_augment(N=1, st="V", io=True)
    ace.bind_equal({"aV", "V"}, constr="aV")
    return ace


def current_source1():
    ace = electrical1port()
    ace.states_augment(N=1, st="I", io=True)
    ace.bind_sum({"aI": 1, "I": 1}, constr="aI")
    return ace


def voltage_source2():
    ace = electrical2port()
    ace.states_augment(N=1, st="V", io=True)
    ace.bind_sum({"aV": 1, "bV": -1, "V": 1}, constr="V")
    ace.bind_sum({"aI", "bI"}, constr="I")
    return ace


def current_source2():
    ace = electrical2port()
    ace.states_augment(N=1, st="I", io=True)
    ace.bind_equal({"aV", "bV"}, constr="V")
    ace.bind_sum({"aI": 1, "bI": 1, "I": 1}, constr="I")
    return ace
