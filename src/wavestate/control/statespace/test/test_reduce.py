"""
"""
from __future__ import division, print_function, unicode_literals
import scipy
import scipy.signal

from transient.statespace import ACE

from wavestate.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot, fpath_join,
)


def test_reduce_ladder(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N = 1, st = 'A', io = True)
    ace.states_augment(N = 1, st = 'B', io = True)
    ace.states_augment(N = 1, st = 'C', io = True)
    ace.states_augment(N = 1, st = 'D', io = True)
    ace.states_augment(N = 1, st = 'E', io = True)
    ace.states_augment(N = 1, st = 'F', io = True)

    ace.bind_equal(['A', 'B', 'C', 'D', 'E', 'F'])

    ace.debug_sparsity_print()
    sccs = ace.strongly_connected_components_reducible(st_start = 'A')
    print('again with C')
    sccs = ace.strongly_connected_components_reducible(st_start = 'C')
    return


def test_reduce_double(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N = 1, st = 'A', io = True)
    ace.states_augment(N = 1, st = 'B', io = True)
    ace.states_augment(N = 1, st = 'C', io = True)
    ace.states_augment(N = 1, st = 'D', io = True)
    ace.states_augment(N = 1, st = 'E', io = True)
    ace.states_augment(N = 1, st = 'F', io = True)

    ace.bind_equal(['A', 'B', 'C', 'E'])
    ace.bind_equal(['D', 'E'])
    ace.bind_equal(['E', 'F'])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start = 'A')
    return

def test_reduce_loop(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N = 1, st = 'A', io = True)
    ace.states_augment(N = 1, st = 'B', io = True)
    ace.states_augment(N = 1, st = 'C', io = True)
    ace.states_augment(N = 1, st = 'D', io = True)
    ace.states_augment(N = 1, st = 'E', io = True)
    ace.states_augment(N = 1, st = 'F', io = True)

    ace.bind_equal(['A', 'B', 'C', 'D'])
    ace.bind_sum(['C', 'D', 'E'])
    ace.bind_equal(['E', 'F'])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start = 'A')
    print('again with B')
    sccs = ace.strongly_connected_components_reducible(st_start = 'B')
    print('again with C')
    sccs = ace.strongly_connected_components_reducible(st_start = 'C')
    return

def test_reduce_loop2(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()
    ace.states_augment(N = 1, st = 'A', io = True)
    ace.states_augment(N = 1, st = 'B', io = True)
    ace.states_augment(N = 1, st = 'C', io = True)
    ace.states_augment(N = 1, st = 'D', io = True)
    ace.states_augment(N = 1, st = 'E', io = True)
    ace.states_augment(N = 1, st = 'F', io = True)

    ace.bind_equal(['A', 'B'])
    ace.bind_equal(['B', 'C'])
    ace.bind_sum(['B', 'C', 'D'])
    ace.bind_sum(['C', 'D', 'E'])
    ace.bind_equal(['E', 'F'])
    ace.debug_sparsity_print()

    sccs = ace.strongly_connected_components_reducible(st_start = 'A')
    print('again with B')
    sccs = ace.strongly_connected_components_reducible(st_start = 'B')
    print('again with C')
    sccs = ace.strongly_connected_components_reducible(st_start = 'C')
    return
