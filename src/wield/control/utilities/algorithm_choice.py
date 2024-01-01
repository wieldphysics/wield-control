#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023 California Institute of Technology.
# SPDX-FileCopyrightText: © 2023 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
This module includes the set of helper functions to maintain the algorithm ranking lists for controls classes.

Even though there is typically a preferred algorithm for many actions, the helper list is useful for comparing
between the performance of different algorithms and implementations.
"""
import warnings


"""
Algorithm choices nested dictionary. This is a mapping of various kinds of algorithms needed by the controls classes.
The outer dictionary is keyed by the names of the algorithms. The inner dictionary is keyed by the text names of the algorithms.
It maps to the priority ranking of the algorithm.
"""
algorithm_choices_defaults = {
    "ss2poles": {
    },
    "ss2zpk": {
    },
    "zpk2ss": {
    },
    "ssreduce": {
    },
    "feedback": {
    },
    "zpk2fresponse": {
    },
    "ss2fresponse": {
    },
    "ss2domainpts": {
    },
    "zpk2domainpts": {
    },
    "domainpts_join": {
    },
    "print_nonzero": {
    },
}

"""
This is a mapping of text names to algorithm functions
"""
algorithm_mappings_name2func = {
}
"""
This is a mapping of algorithm functions to text names
"""
algorithm_mappings_func2name = {}
"""
This is a mapping of algorithm functions to default rankings
"""
algorithm_mappings_name2rank = {}
algorithm_mappings_name2type = {}


def algorithm_register(typename, textname, func, rank):
    if textname in algorithm_mappings_name2func:
        raise RuntimeError("Overlapping algorithm name registration")
    algorithm_mappings_name2func[textname] = func
    algorithm_mappings_func2name[func] = textname
    algorithm_mappings_name2rank[textname] = func
    algorithm_mappings_name2type[textname] = typename

    submap = algorithm_choices_defaults.setdefault(typename, {})
    submap[textname] = rank
    return


def algo_merge(choicesA, choicesB):
    """
    Takes two algorithm_choices dictionaries and merges them into a single dictionary.
    Raises a warning if rankings end up identical.
    """
    tset = set(choicesA.keys()) | set(choicesB.keys())
    choicesN = {}
    for atype in tset:
        rankN = {}
        choicesN[atype] = rankN
        rankA = choicesA.get(atype, {})
        rankB = choicesB.get(atype, {})
        rset = set(rankA.keys()) | set(rankB.keys())
        for rname in rset:
            rA = rankA.get(rname, None)
            rB = rankB.get(rname, None)
            if rA is None:
                if rB is None:
                    rN = None
                else:
                    rN = rB
            else:
                if rB is None:
                    rN = rA
                elif rA == rB:
                    warnings.warn("Algorithm ranking merge has ambiguous ranking")
                    rN = rA
                elif rA > rB:
                    rN = rA
                else:
                    rN = rB
            rankN[rname] = rN
    return choicesN


def algo_merge_full(choicesA, rankingA, choicesB, rankingB):
    """
    merges choices and generates rankings. If the choices are the same, then less work is done
    """
    if choicesA is choicesB:
        return choicesA, rankingA
    choicesN = algo_merge(choicesA, choicesB)
    rankingN = algo2ranking(choicesN)
    return choicesN, rankingN


def algo2ranking(choices):
    ranks = {}
    for rtype, rdict in choices.items():
        rlist = []
        ranks[rtype] = rlist
        # sort highest to lowest index
        rl = [kv[0] for kv in sorted(rdict.items(), key = lambda kv: (-kv[1], kv[0]))]
        for name in rl:
            func = algorithm_mappings_name2func[name]
            assert (algorithm_mappings_name2type[name] == rtype)
            rlist.append(func)

    return ranks


def choices_and_rankings(algorithm_choices, algorithm_ranking):
    """
    """
    if algorithm_choices is None:
        assert (algorithm_ranking is None)
        algorithm_choices = algorithm_choices_defaults

    if algorithm_ranking is None:
        # merge with the defaults to make a complete set
        algorithm_choices = algo_merge(algorithm_choices, algorithm_choices_defaults)
        algorithm_ranking = algo2ranking(algorithm_choices)

    return algorithm_choices, algorithm_ranking


def algo_run(algotype, rankings, args=(), kwargs={}, choice=None):
    """
    Runs the algorithm functions in order. Exceptions cause a later algorithm to be run

    choice = specifically choose the algorithm to run
    """
    if choice is None:
        ranks = rankings[algotype]
    else:
        func = algorithm_mappings_name2func[choice]
        assert(algorithm_mappings_name2type[choice] == algotype)
        ranks = [func]

    for func in ranks:
        try:
            ret = func(*args, **kwargs)
        except AlgorithmError:
            # hmmm. Probably shouldn't accept any kind of exception here
            continue
        break
    else:
        if not ranks:
            raise RuntimeError("No Algorithm Found for type {}".format(algotype))
        else:
            raise RuntimeError("No algorithm can handle the current properties of your StateSpace")
    return ret


class AlgorithmError(ValueError):
    """
    Error to raise if algorithm is failing
    """
    pass
