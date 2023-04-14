#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import scipy
import scipy.signal

from wield.control.ACE import ACE

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
    test_trigger,
    tpath_preclear,
)


def test_reduce_ladder(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N=1, st="A", io=True)
    ace.states_augment(N=1, st="B", io=True)
    ace.states_augment(N=1, st="C", io=True)
    ace.states_augment(N=1, st="D", io=True)
    ace.states_augment(N=1, st="E", io=True)
    ace.states_augment(N=1, st="F", io=True)

    ace.bind_equal(["A", "B", "C", "D", "E", "F"])

    ace.debug_sparsity_print()
    sccs = ace.strongly_connected_components_reducible(st_start="A")
    print("again with C")
    sccs = ace.strongly_connected_components_reducible(st_start="C")
    return


def test_reduce_double(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N=1, st="A", io=True)
    ace.states_augment(N=1, st="B", io=True)
    ace.states_augment(N=1, st="C", io=True)
    ace.states_augment(N=1, st="D", io=True)
    ace.states_augment(N=1, st="E", io=True)
    ace.states_augment(N=1, st="F", io=True)

    ace.bind_equal(["A", "B", "C", "E"])
    ace.bind_equal(["D", "E"])
    ace.bind_equal(["E", "F"])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start="A")
    return


def test_reduce_loop(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N=1, st="A", io=True)
    ace.states_augment(N=1, st="B", io=True)
    ace.states_augment(N=1, st="C", io=True)
    ace.states_augment(N=1, st="D", io=True)
    ace.states_augment(N=1, st="E", io=True)
    ace.states_augment(N=1, st="F", io=True)

    ace.bind_equal(["A", "B", "C", "D"])
    ace.bind_sum(["C", "D", "E"])
    ace.bind_equal(["E", "F"])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start="A")
    print("again with B")
    sccs = ace.strongly_connected_components_reducible(st_start="B")
    print("again with C")
    sccs = ace.strongly_connected_components_reducible(st_start="C")
    return


def test_reduce_loop2(dprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N=1, st="A", io=True)
    ace.states_augment(N=1, st="B", io=True)
    ace.states_augment(N=1, st="C", io=True)
    ace.states_augment(N=1, st="D", io=True)
    ace.states_augment(N=1, st="E", io=True)
    ace.states_augment(N=1, st="F", io=True)

    ace.bind_equal(["A", "B"])
    ace.bind_equal(["B", "C"])
    ace.bind_sum(["B", "C", "D"])
    ace.bind_sum(["C", "D", "E"])
    ace.bind_equal(["E", "F"])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start="A")
    print("again with B")
    sccs = ace.strongly_connected_components_reducible(st_start="B")
    print("again with C")
    sccs = ace.strongly_connected_components_reducible(st_start="C")
    return
