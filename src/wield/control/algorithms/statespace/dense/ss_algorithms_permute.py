#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import collections
import numpy as np
from wield.bunch import Bunch


TupleABCDE = collections.namedtuple("ABCDE", ('A', 'B', 'C', 'D', 'E'))


def permute_UT(A, B, C, D, E, orientation='upper'):
    """
    Permute elements to make the A and E matrix upper-triangular dominant.

    This algorithm is for TESTING. It does not appear to help stability.
    """
    A_orig = A.copy()
    # return A,B,C,D,E
    edges_consider = []
    edges_use = []
    # STATES:
    # 0: unchecked
    # 1: used
    # -1: tossed
    # -2: tossed (greedy)

    Nstates = A.shape[-1]
    Nconstr = A.shape[-2]
    # A = np.eye(N=Nstates, k=1)

    # this is a map indexed by FROM (column), holding edges TO
    emap = []

    # requires square for now
    assert (Nstates == Nconstr)
    for irow in range(Nconstr):
        emap.append([])
        for icol in range(Nstates):
            # ignore diagonals
            if irow == icol:
                continue
            val = A[irow, icol]

            # also don't consider null elements
            if val == 0:
                continue

            eB = Bunch(
                aval = abs(val),
                val = val,
                irow = irow,
                icol = icol,
                state = 0,
            )
            edges_consider.append(eB)

    # highest priority edges_consider are on TOP, so that we can pop them off
    edges_consider = sorted(edges_consider, key = lambda eB: eB.aval)

    # higher rank is higher priority
    # this ensures they are inserted in sorted order
    for idx, eB in enumerate(edges_consider):
        eB.rank = idx
        emap[eB.icol].append(eB)

    # now try to sort

    stateset = np.zeros(Nstates, dtype=bool)
    statestack = []

    def depth_topo_greedy(current_inode):
        current_edges = emap[current_inode]
        print("CURRENT: ", current_inode)
        print(statestack)

        # iterate DOWN from the highest priority edges
        for idx_edge in range(len(current_edges)-1, -1, -1):

            eB = current_edges[idx_edge]

            # has it already been visited?
            if not stateset[eB.irow]:
                statestack.append(eB.irow)
                stateset[eB.irow] = True
                # now go down
                ret = depth_topo_greedy(eB.irow)
                # undo the states and stack
                statestack.pop()
                stateset[eB.irow] = False

                if not ret:
                    # cycle found in a lower element.
                    # so consider if this one needs to be removed
                    do_remove = True
                else:
                    do_remove = False
            else:
                do_remove = True

            if do_remove:
                # ok, already visited, this means a loop
                # now we toss out depending on the state
                if eB.state == 0:
                    # edge hasn't been considered, and can be tossed out
                    eB.state = -2
                    current_edges.pop(idx_edge)
                    # move on to the next edge
                    continue
                elif eB.state < 0:
                    raise RuntimeError("Algorithm Error, edge should already be removed")
                else:
                    # must be an accepted edge causing the cycle, so an upstream edge must have an error.
                    return False
        # it got through all edges, yay!
        # return True to indicate a successful sort
        return True

    while edges_consider:
        # now run it on the top-most edge
        print("REMAINING: ", len(edges_consider))
        top_eB = edges_consider.pop()
        if top_eB.state < 0:
            # already removed edge
            continue

        statestack.append(top_eB.icol)
        stateset[top_eB.icol] = True
        ret = depth_topo_greedy(top_eB.irow)
        statestack.pop()
        stateset[top_eB.icol] = False

        # check that the state is reset
        assert (np.all(~stateset))
        assert (not statestack)

        # successful topo sort after greedy removal, so accept the edge
        if ret:
            top_eB.state = 2
            edges_use.append(top_eB)
        else:
            # cycle was forced, so we cannot accept this edge
            top_eB.state = -1
            emap[top_eB.icol].remove(top_eB)

    # find the root edges to start the topo sort
    for eB in edges_use:
        stateset[eB.irow] = True
    root_edges = np.argwhere(~stateset)[:, 0][::-1]
    print("ROOTS: ", root_edges)
    
    stateset[:] = False
    def depth_topo_final(current_inode):
        stateset[current_inode] = True
        current_edges = emap[current_inode]

        # iterate DOWN from the highest priority edges
        for idx_edge in range(len(current_edges)-1, -1, -1):
            eB = current_edges[idx_edge]
            # only append and recurse if not already visited
            if not stateset[eB.irow]:
                statestack.append(eB.irow)
                # now go down
                depth_topo_final(eB.irow)
    for iroot in root_edges:
        statestack.append(iroot)
        depth_topo_final(iroot)

    # statestack = statestack[::-1]

    print("ORDER: ", statestack)
    assert(len(statestack) == Nstates)
    print("ORDERED", sorted(statestack))

    # now APPLY PERMUTATION

    A = A_orig.copy()
    P = np.asarray(statestack)
    # P = np.arange(Nstates)[::-1]

    def invert_permutation(p):
        """Return an array s with which np.array_equal(arr[p][s], arr) is True.
        The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.

        from https://stackoverflow.com/a/25535723
        """
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    Pi = invert_permutation(P)
    P2 = invert_permutation(Pi)
    assert np.all(P == P2)
    # Pi, P = P, Pi
    # do we need to bother?

    Ap = A.copy()[..., P, :][..., :, P]

    if E is not None:
        Ep = E.copy()[..., P, :][..., :, P]
    else:
        Ep = E

    Bp = B.copy()[..., P, :]
    Cp = C.copy()[..., :, P]

    return TupleABCDE(Ap, Bp, Cp, D, Ep)


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.

    from https://stackoverflow.com/a/25535723
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    # Pi = invert_permutation(P)
    return s
