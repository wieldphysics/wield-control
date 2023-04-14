#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

from collections import defaultdict
import warnings


def SRE_matrix_mult(sreL, sreR):
    seqL, reqL, edges_L = sreL
    seqR, reqR, edges_R = sreR

    seq = dict()
    req = dict()
    edges = dict()

    for k_mid, mseq in seqR.items():
        if not mseq:
            continue
        for k_from in reqL[k_mid]:
            eL = edges_L[k_from, k_mid]
            seq.setdefault(k_from, set())
            for k_to in mseq:
                if k_to not in seqR[k_mid]:
                    continue
                eR = edges_R[k_mid, k_to]
                if k_to in seq[k_from]:
                    edges[k_from, k_to] = edges[k_from, k_to] + eL @ eR
                else:
                    edges[k_from, k_to] = eL @ eR
                    seq[k_from].add(k_to)
                    req.setdefault(k_to, set()).add(k_from)

    if __debug__:
        SRE_check((seq, req, edges))
    return seq, req, edges


def SRE_check(sre):
    seq, req, edges = sre
    for node in req:
        for rnode in req[node]:
            assert node in seq[rnode]
    for node in seq:
        for snode in seq[node]:
            assert node in req[snode]
            edges[node, snode]


def dictset_copy(d):
    d2 = defaultdict(set)
    for k, s in d.items():
        d2[k] = set(s)
    return d2


def SRE_copy(sre):
    Oseq, Oreq, Oedges = sre
    seq = dictset_copy(Oseq)
    req = dictset_copy(Oreq)
    edges = dict(Oedges)
    return seq, req, edges


def check_seq_req_balance(seq, req, edges=None):
    for node, seq_set in seq.items():
        for snode in seq_set:
            assert node in req[snode]
            if edges and (node, snode) not in edges:
                warnings.warn(repr((node, snode)) + "not in edge map")
                edges[node, snode] = 0

    for node, req_set in req.items():
        for rnode in req_set:
            assert node in seq[rnode]


def color_purge_inplace(
    start_set,
    emap,
    seq,
    req,
    edges,
):
    # can't actually purge, must color all nodes
    # from the exception set and then subtract the
    # remainder.
    # purging algorithms otherwise have to deal with
    # strongly connected components, which makes them
    # no better than coloring
    active_set = set()
    active_set_pending = set()
    # print("PURGE START: ", start_set)
    for node in start_set:
        active_set_pending.add(node)

    while active_set_pending:
        node = active_set_pending.pop()
        # print("PURGE NODE: ", node)
        active_set.add(node)
        for snode in emap[node]:
            if snode not in active_set:
                active_set_pending.add(snode)
    full_set = set(seq.keys())
    purge_set = full_set - active_set
    # print("FULL_SET", active_set)
    # print("PURGE", len(purge_set), len(full_set))
    purge_subgraph_inplace(seq, req, edges, purge_set)


def purge_reqless_inplace(
    seq,
    req,
    except_set=None,
    req_alpha=None,
    seq_beta=None,
    edges=None,
):
    if except_set is None:
        except_set = set(req_alpha.keys())
    color_purge_inplace(
        except_set,
        seq,
        seq,
        req,
        edges,
    )
    if seq_beta is not None:
        rmnodes = list()
        for node in seq_beta.keys():
            if not req.get(node, None):
                rmnodes.append(node)
        for node in rmnodes:
            del seq_beta[node]


def purge_seqless_inplace(
    seq,
    req,
    except_set=None,
    req_alpha=None,
    seq_beta=None,
    edges=None,
):
    if except_set is None:
        except_set = set(seq_beta.keys())
    color_purge_inplace(
        except_set,
        req,
        seq,
        req,
        edges,
    )
    if req_alpha is not None:
        rmnodes = list()
        for node in req_alpha.keys():
            if not seq.get(node, None):
                rmnodes.append(node)
        for node in rmnodes:
            del req_alpha[node]


def purge_inplace(
    seq,
    req,
    req_alpha,
    seq_beta,
    edges,
):
    purge_reqless_inplace(
        seq=seq,
        req=req,
        seq_beta=seq_beta,
        req_alpha=req_alpha,
        edges=edges,
    )
    purge_seqless_inplace(
        seq=seq,
        req=req,
        seq_beta=seq_beta,
        req_alpha=req_alpha,
        edges=edges,
    )
    return


def edgedelwarn(
    edges,
    nfrom,
    nto,
):
    if edges is None:
        return
    try:
        del edges[nfrom, nto]
    except KeyError:
        warnings.warn(repr(("Missing edge", nfrom, nto)))


def purge_subgraph_inplace(
    seq,
    req,
    edges,
    purge_set,
):
    for node in purge_set:
        for snode in seq[node]:
            edgedelwarn(edges, node, snode)
            if snode not in purge_set and (snode, node):
                req[snode].remove(node)
        del seq[node]
        for rnode in req[node]:
            # edgedelwarn(edges, rnode, node)
            if rnode not in purge_set and (rnode, node):
                seq[rnode].remove(node)
        del req[node]
    return


def pre_purge_inplace(seq, req, edges):
    # print("PRE-PURGING")
    total_N = 0
    purge_N = 0
    # actually needs to list this as seq is mutating
    for inode, smap in list(seq.items()):
        for snode in list(smap):
            total_N += 1
            if (inode, snode) not in edges:
                # if purge_N % 100:
                #    print("DEL: ", inode, snode)
                purge_N += 1
                smap.remove(snode)
    for snode, rmap in req.items():
        for inode in list(rmap):
            if (inode, snode) not in edges:
                rmap.remove(inode)
    # print("FRAC REMOVED: ", purge_N / total_N, purge_N)


def SRABE_copy(SRE):
    seq, req, req_alpha, seq_beta, edges = SRE

    edges2 = dict()

    seq2 = defaultdict(set)
    for node, sset in seq.items():
        seq2[node].update(sset)
        for n_to in sset:
            edges2[node, n_to] = edges[node, n_to]
    req2 = defaultdict(set)
    for node, rset in req.items():
        req2[node].update(rset)

    seq_beta2 = defaultdict(set)
    for node, sset in seq_beta2.items():
        seq_beta2[node].update(sset)
        for n_to in sset:
            edges2[node, n_to] = edges[node, n_to]

    req_alpha2 = defaultdict(set)
    for node, rset in req_alpha.items():
        req_alpha2[node].update(rset)
        for n_fr in sset:
            edges2[n_fr, node] = edges[n_fr, node]

    return seq2, req2, req_alpha2, seq_beta2, edges2
