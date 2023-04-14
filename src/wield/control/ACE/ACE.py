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
import numbers

from wield.bunch import Bunch
from collections import defaultdict, Mapping
from wield.utilities.np import broadcast_shapes

from .tupleize import tupleize


class ACE(object):
    def __init__(self):
        self.As = dict()

        self.Es = dict()
        # indicates the known rank of E matrix 0 or missing means 0 rank. -1 means full rank (min of row or col rank)
        # None means that the rank is unknown or a numerical rank.
        self.Eranks = dict()

        self.Cs = dict()

        # matrix into Cs to add more states, but make it more obvious that
        # controllability and observability are maintained by not adding excessive Cs requiring later reductions
        # this is in (io, xo) pairs, as though it multiplies Cs
        self.Xs = dict()

        # mapping of states to Ainv matrices, used to construct D matrices
        self.AinvDcr = dict()

        # collection of noise names to a pair of sets. The first are io names
        # and the second are other noises.
        self.noises = dict()
        # the map of io/xo into binding constraints
        self.bound_io = dict()

        # there are labels for {states, constr, output}
        self.st2crA = defaultdict(set)
        self.cr2stA = defaultdict(set)
        self.st2crE = defaultdict(set)
        self.cr2stE = defaultdict(set)
        self.st2io  = defaultdict(set)
        self.io2st  = defaultdict(set)
        # for the Xs matrix
        # this is in C2io to Cio edges
        self.xo2io = defaultdict(set)

        self.st = defaultdict(Bunch)
        self.cr = defaultdict(Bunch)
        # also contains the xo's, as they are a form (2nd layer) of io
        self.io = defaultdict(Bunch)

        # mapping of port names to Bunches, containing the individual xo's to bind
        # flow types contain a .I = xo, and .O = xo. Conserved quantities are in
        # .potential = xo and .flow = xo
        self.pt = dict()
        return

    def copy(self):
        return copy.deepcopy(self)

    def __copy__(self):
        return copy.deepcopy(self)

    @classmethod
    def from_ABCD(
        cls,
        A,
        B,
        C,
        D,
        E=None,
        cmn=None,
        states="S",
        constr="C",
        output="O",
        inputs="I",
        instates=None,
    ):
        ace = cls()
        if instates is None:
            instates = inputs
        ace.statespace_add(
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
            cmn=cmn,
            states=states,
            constr=constr,
            output=output,
            inputs=inputs,
            instates=instates,
        )
        return ace

    def statespace_add(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        E=None,
        cmn=None,
        states=None,
        constr=None,
        output=None,
        inputs=None,
        instates=None,
    ):
        if constr is None:
            if A is None:
                constr = inputs
            else:
                constr = states

        if A is not None or E is not None or B is not None:
            assert states is not None

        if A is not None or E is not None or B is not None:
            assert constr is not None

        if B is not None or D is not None:
            assert inputs is not None

        if C is not None or D is not None:
            assert output is not None

        cmn = tupleize(cmn)
        states = cmn + tupleize(states)
        constr = cmn + tupleize(constr)
        output = cmn + tupleize(output)
        inputs = cmn + tupleize(inputs)
        if instates is None:
            instates = inputs
        else:
            instates = cmn + tupleize(instates)

        N_states = None
        N_constr = None
        N_output = None
        N_inputs = None
        N_instates = None

        if states in self.st:
            N_states = self.st[states].N
        if instates in self.st:
            N_instates = self.st[instates].N
        if constr in self.cr:
            N_constr = self.cr[constr].N
        if output in self.io:
            N_output = self.io[output].N
        if inputs in self.io:
            N_inputs = self.io[inputs].N

        if A is not None:
            assert constr not in self.st2crA[states]
            self.st2crA[states].add(constr)

            assert states not in self.cr2stA[constr]
            self.cr2stA[constr].add(states)

            assert N_states is None or A.shape[-1] == N_states
            assert N_constr is None or A.shape[-2] == N_constr
            N_states = A.shape[-1]
            N_constr = A.shape[-2]

            assert (states, constr) not in self.As
            self.As[states, constr] = A

            if E is None or E is 1:
                if N_states != N_constr:
                    raise RuntimeError(
                        "Cannot assume E matrix for rank-deficient system"
                    )
                # TODO, this won't work with A having extra shape
                self.Es[states, constr] = np.eye(N_states)
                self.Eranks[states, constr] = -1

                self.st2crE[states].add(constr)
                self.cr2stE[constr].add(states)
            elif E is 0:
                # not inserting it counts as a null
                pass
            else:
                self.Es[states, constr] = E
                # TODO, this could be separated logic
                if E.shape[-2] == E.shape[-1]:
                    if np.all(E == np.eye(E.shape[-2])):
                        self.Eranks[states, constr] = -1
                    else:
                        self.Eranks[states, constr] = None
                else:
                    # TODO, allow Erank to be specified
                    self.Eranks[states, constr] = None
                self.st2crE[states].add(constr)
                self.cr2stE[constr].add(states)

                assert N_states is None or E.shape[-1] == N_states
                assert N_constr is None or E.shape[-2] == N_constr

        if C is not None:
            assert output not in self.st2io[states]
            self.st2io[states].add(output)

            assert states not in self.io2st[output]
            self.io2st[output].add(states)

            assert N_states is None or C.shape[-1] == N_states
            assert N_output is None or C.shape[-2] == N_output
            N_states = C.shape[-1]
            N_output = C.shape[-2]

            assert (states, output) not in self.Cs
            self.Cs[states, output] = C

        if B is not None:
            assert inputs not in self.st2io[instates]
            self.st2io[instates].add(inputs)

            assert instates not in self.io2st[inputs]
            self.io2st[inputs].add(instates)

            assert constr not in self.st2crA[instates]
            self.st2crA[instates].add(constr)

            assert instates not in self.cr2stA[constr]
            self.cr2stA[constr].add(instates)

            assert N_instates is None or B.shape[-1] == N_instates
            assert N_inputs is None or B.shape[-1] == N_inputs
            assert N_constr is None or B.shape[-2] == N_constr
            N_instates = B.shape[-1]
            N_inputs = B.shape[-1]
            N_constr = B.shape[-2]

            assert (instates, inputs) not in self.Cs
            self.Cs[instates, inputs] = np.eye(N_inputs)
            assert (instates, constr) not in self.As
            self.As[instates, constr] = B
            assert (instates, constr) not in self.Es
            # no need to specify the default of 0
            # self.Es[instates, constr] = np.zeros((N_constr, N_inputs))
            # self.Eranks[instates, constr] = 0

        if np.all(D == 0):
            D = None
        if D is not None:
            assert output not in self.st2io[instates]
            self.st2io[instates].add(output)

            assert instates not in self.io2st[output]
            self.io2st[output].add(instates)

            assert N_instates is None or D.shape[-1] == N_instates
            assert N_inputs is None or D.shape[-1] == N_inputs
            assert N_output is None or D.shape[-2] == N_output
            N_instates = D.shape[-1]
            N_inputs = D.shape[-1]
            N_output = D.shape[-2]

            assert (instates, output) not in self.Cs
            self.Cs[instates, output] = D
            if (instates, inputs) not in self.Cs:
                self.Cs[instates, inputs] = np.eye(N_inputs)
            else:
                assert np.all(self.Cs[instates, inputs] == np.eye(N_inputs))

        # activate the dict in the defaultdicts
        if N_states is not None:
            self.st[states].N = N_states
        if N_instates is not None:
            self.st[instates].N = N_instates
        if N_constr is not None:
            self.cr[constr].N = N_constr
        if N_output is not None:
            self.io[output].N = N_output
        if N_inputs is not None:
            self.io[inputs].N = N_inputs
        return

    def insert(self, ace, cmn=None):
        """
        Insert the ACE system into this one. Add the cmn factor to all of the names:

        TODO, add xo2io and Xs
        """
        st_map = dict()
        cr_map = dict()
        io_map = dict()
        pt_map = dict()
        cmn = tupleize(cmn)

        def anno_update(xx_map, self_xx, ace_xx):
            for xx, xx_ace in ace_xx.items():
                xx_new = xx_map[xx] = cmn + xx
                xx_ace = copy.deepcopy(xx_ace)
                xx_prev = self_xx.setdefault(xx_new, xx_ace)
                if xx_prev != xx_ace:
                    # now need to check and update that the cmn annotations are identical
                    xx_cmn = set(xx_prev) & set(xx_ace)
                    xx_anno_ace = {anno: xx_ace[anno] for anno in xx_cmn}
                    xx_anno_self = {anno: xx_prev[anno] for anno in xx_cmn}
                    assert xx_anno_ace == xx_anno_self

        anno_update(st_map, self.st, ace.st)
        anno_update(cr_map, self.cr, ace.cr)
        anno_update(io_map, self.io, ace.io)

        def port_update(xx_map, self_xx, ace_xx):
            for xx, xx_ace in ace_xx.items():
                xx_new = xx_map[xx] = cmn + xx
                xx_ace = copy.deepcopy(xx_ace)
                for pname in ["potential", "flow", "I", "O"]:
                    if pname in xx_ace:
                        xx_ace[pname] = io_map[xx_ace[pname]]
                assert xx_new not in self_xx
                self_xx[xx_new] = xx_ace

        port_update(pt_map, self.pt, ace.pt)

        for st, cr_set in ace.st2crA.items():
            st_new = st_map[st]
            self.st2crA[st_new].update(cr_map[cr] for cr in cr_set)

        for cr, st_set in ace.cr2stA.items():
            cr_new = cr_map[cr]
            self.cr2stA[cr_new].update(st_map[st] for st in st_set)

        for st, cr_set in ace.st2crE.items():
            st_new = st_map[st]
            self.st2crE[st_new].update(cr_map[cr] for cr in cr_set)

        for cr, st_set in ace.cr2stE.items():
            cr_new = cr_map[cr]
            self.cr2stE[cr_new].update(st_map[st] for st in st_set)

        for st, io_set in ace.st2io.items():
            st_new = st_map[st]
            self.st2io[st_new].update(io_map[io] for io in io_set)

        for io, st_set in ace.io2st.items():
            io_new = io_map[io]
            self.io2st[io_new].update(st_map[st] for st in st_set)

        for xo, io_set in ace.xo2io.items():
            xo_new = io_map[xo]
            self.xo2io[xo_new].update(io_map[io] for io in io_set)

        for (st, cr), mat in ace.As.items():
            st_new = st_map[st]
            cr_new = cr_map[cr]
            mat_prev = self.As.setdefault((st_new, cr_new), mat)
            assert mat_prev is mat

        for (st, cr), mat in ace.Es.items():
            st_new = st_map[st]
            cr_new = cr_map[cr]
            mat_prev = self.Es.setdefault((st_new, cr_new), mat)

            # self.Eranks[instates, constr] = 0
            assert mat_prev is mat
            Erank = ace.Eranks.get((st, cr), 0)
            if Erank != 0:
                self.Eranks.setdefault((st_new, cr_new), Erank)

        for (st, io), mat in ace.Cs.items():
            st_new = st_map[st]
            io_new = io_map[io]
            mat_prev = self.Cs.setdefault((st_new, io_new), mat)
            assert mat_prev is mat

        for (io1, xo2), mat_ish in ace.Xs.items():
            io1_new = io_map[io1]
            xo2_new = io_map[xo2]
            mat_ish_prev = self.Xs.setdefault((io1_new, xo2_new), mat_ish)
            assert mat_ish_prev is mat_ish

        # finally, do the noises
        for n, (io_set, n_set, cats) in ace.noises.items():
            n = cmn + n
            io_set = {io_map[io] for io in io_set}
            n_set = {cmn + n for n in n_set}
            assert n not in self.noises
            self.noises[n] = (io_set, n_set, set(cats))
            for cat in cats:
                self.cat2noise[cat].add(n)

        for cr, AnI in ace.AinvDcr.items():
            cr_new = cr_map[cr]
            self.AinvDcr[cr_new] = AnI

        return

    def port_add_conserved(
        self,
        pt,
        type,
        flow=None,
        potential=None,
    ):
        pt = tupleize(pt)
        pB = self.pt.setdefault(pt, Bunch())
        if type is not None:
            pB.type = type

        if flow is not None or potential is not None:
            assert flow is not None
            assert potential is not None
            flow = tupleize(flow)
            potential = tupleize(potential)
            assert flow in self.io
            assert potential in self.io
            pB.flow = flow
            pB.potential = potential
        return

    def port_add_scatter(
        self,
        pt,
        type,
        In=None,
        Out=None,
        impedance=None,
    ):
        pt = tupleize(pt)
        pB = self.pt.setdefault(pt, Bunch())

        if type is not None:
            pB.type = type

        if In is not None or Out is not None:
            assert In is not None
            assert Out is not None
            In = tupleize(In)
            Out = tupleize(Out)
            assert In in self.io
            assert Out in self.io
            pB.In = In
            pB.Out = Out
            if impedance is not None:
                self.impedance = impedance
        return

    def noise_add(self, noise, io_set=(), noise_set=(), categories=()):
        noise = tupleize(noise)
        io_set = set(tupleize(io) for io in io_set)
        noise_set = set(tupleize(n) for n in noise_set)
        categories = set(categories)

        for io in io_set:
            assert io in self.bound_io

        self.noises[noise] = (io_set, noise_set, categories)
        for cat in categories:
            self.cat2noise[cat].add(noise)
        return

    def states_augment(self, N, st, io=None):
        st = tupleize(st)

        assert st not in self.st
        self.st[st].N = N
        if io is not None:
            if io is True:
                io = st
            else:
                io = tupleize(io)
            assert io not in self.io
            self.io[io].N = N

            assert io not in self.st2io[st]
            self.st2io[st].add(io)

            assert st not in self.io2st[io]
            self.io2st[io].add(st)

            assert (st, io) not in self.Cs
            self.Cs[st, io] = np.eye(N)

        return

    def io_input(self, io, constr=None):
        xo = tupleize(io)
        # add a zero binding
        if constr is None:
            constr = xo
        else:
            constr = tupleize(constr)
        assert xo in self.io
        # now add a noise constraint associated with the xo
        self.bound_io[xo] = constr
        Cs, N = self._Cs_collect(xo)
        assert constr not in self.cr
        self.cr[constr].N = N
        assert N > 0
        for st, C in Cs.items():
            self.As[st, constr] = C
            self.st2crA[st].add(constr)
            self.cr2stA[constr].add(st)
        return

    def io_add(self, io, matmap, constr=None):
        """
        Adds secondary outputs, remapping the existing outputs.
        matmap is a mapping of existing outputs to matrices to sum across.

        the mappings can be either a list of indices, a matrix, or a (matrix, index) tuple

        if constr is not None, then add a single binding constraint and consider it a noise IO.
        """
        xo = tupleize(io)
        N = None

        # this is the fundamental C mapping, mapped through the Xs
        Cmapping = dict()

        for io2, mat_ish in matmap.items():
            io2 = tupleize(io2)
            ioset = self.xo2io.get(io2, None)
            scalar = None
            if mat_ish is None:
                # is a pass-through
                idx_ish = None
                mat_ish = None
            elif isinstance(mat_ish, numbers.Number):
                scalar = mat_ish
                idx_ish = None
                mat_ish = None
            elif isinstance(mat_ish, list):
                # is an index mapping
                idx_ish = np.array(mat_ish)
                mat_ish = None
                N = len(idx_ish)
            elif isinstance(mat_ish, tuple):
                tup_ish = mat_ish
                if len(tup_ish) == 0:
                    idx_ish = None
                else:
                    idx_ish = tup_ish[-1]

                if len(tup_ish) <= 1:
                    mat_ish = None
                elif len(tup_ish) >= 1:
                    mat_ish = tup_ish[-2]
            else:
                mat_ish = mat_ish
                idx_ish = None

            if ioset is None:
                if mat_ish is not None:
                    N = mat_ish.shape[-2]
                elif idx_ish is not None:
                    N = len(idx_ish)
                else:
                    N = self.io[io2].N
                if scalar is not None:
                    if mat_ish is not None:
                        mat_ish = scalar * mat_ish
                    else:
                        mat_ish = scalar * np.eye(N)
                Cmapping[io2] = (mat_ish, idx_ish)
            else:
                for io3 in ioset:
                    c2 = self.Xs[io3, io2]
                    mat_ish2, idx_ish2 = c2
                    if idx_ish is not None:
                        if mat_ish2 is not None:
                            if mat_ish is not None:
                                mat_ish = mat_ish @ mat_ish2[idx_ish, :]
                                idx_ish = idx_ish2
                            else:
                                mat_ish = mat_ish2[idx_ish, :]
                                idx_ish = idx_ish2
                        else:
                            # mat_ish doesn't depend on this
                            if idx_ish2 is not None:
                                idx_ish = idx_ish2[idx_ish2]
                            else:
                                idx_ish = idx_ish2
                    else:
                        if mat_ish2 is not None:
                            idx_ish = idx_ish2
                            if mat_ish is not None:
                                mat_ish = mat_ish @ mat_ish2
                            else:
                                mat_ish = mat_ish2
                    if scalar is not None:
                        if mat_ish is not None:
                            mat_ish = scalar * mat_ish
                        else:
                            if idx_ish is not None:
                                N = len(idx_ish)
                            else:
                                N = self.io[io3].N
                            mat_ish = scalar * np.eye(N)
                    Cmapping[io3] = (mat_ish, idx_ish)

                # get N from the last io3 mapping
                if mat_ish is not None:
                    N = mat_ish.shape[-2]
                elif idx_ish is not None:
                    N = len(idx_ish)
                else:
                    N = self.io[io3].N

        assert N is not None
        self.io[xo].N = N

        for io2, C in Cmapping.items():
            self.xo2io[xo].add(io2)
            self.Xs[io2, xo] = C

        if constr is not None:
            if constr is True:
                constr = None
            self.io_input(io=xo, constr=constr)
        return

    def _Cs_collect(self, xotup, mat=1, _AinvDs=False):
        """
        Collects C matrices out of both self.Cs and self.Xs. The collection is
        stored in a dictionary, which sums the C matrices if needed.
        """
        ioset = self.xo2io.get(xotup, None)
        dictCs = dict()

        if mat is None:
            mat = 1

        def update_Cs(st, c_new):
            l_prev = len(dictCs)
            c_old = dictCs.setdefault(st, c_new)
            if l_prev == len(dictCs):
                dictCs[st] = c_old + c_new

        if not _AinvDs:
            io2st = self.io2st
        else:
            io2st = self.io2stD

        N = None
        if ioset is None:
            if mat is 1:
                for st in io2st[xotup]:
                    c = self.Cs[st, xotup]
                    if N is None:
                        N = c.shape[-2]
                    else:
                        assert N == c.shape[-2]
                    update_Cs(st, c)
            elif mat is -1:
                for st in io2st[xotup]:
                    c = self.Cs[st, xotup]
                    if N is None:
                        N = c.shape[-2]
                    else:
                        assert N == c.shape[-2]
                    update_Cs(st, -c)
            else:
                N = mat.shape[-2]
                for st in io2st[xotup]:
                    c = self.Cs[st, xotup]
                    update_Cs(st, mat @ c)
        else:
            for io2 in ioset:
                c2 = self.Xs[io2, xotup]
                mat_ish, idx_ish = c2
                if idx_ish is None:
                    idx_ish = slice(None, None, None)

                if mat_ish is not None:
                    if mat is 1:
                        pass
                    elif isinstance(mat, numbers.Number) or isinstance(mat_ish, numbers.Number):
                        mat_ish = mat * mat_ish
                    else:
                        mat_ish = mat @ mat_ish
                else:
                    mat_ish = mat

                if mat_ish is 1:
                    for st in io2st[io2]:
                        c = self.Cs[st, io2]
                        c_new = c[..., idx_ish, :]
                        N2 = c_new.shape[-2]
                        if N is None:
                            N = N2
                        else:
                            assert N == N2
                        update_Cs(st, c_new)
                elif isinstance(mat_ish, numbers.Number):
                    for st in io2st[io2]:
                        c = self.Cs[st, io2]
                        c_new = -c[..., idx_ish, :]
                        N2 = c_new.shape[-2]
                        if N is None:
                            N = N2
                        else:
                            assert N == N2
                        update_Cs(st, c_new)
                else:
                    for st in io2st[io2]:
                        c = self.Cs[st, io2]
                        update_Cs(st, mat_ish @ c[..., idx_ish, :])
                    N2 = mat_ish.shape[-2]
                    if N is None:
                        N = N2
                    else:
                        assert N == N2
        if not dictCs:
            N = 0
        return dictCs, N

    def bind_equal(self, set_or_dict, constr=None):
        return self.bind("equal", set_or_dict=set_or_dict, constr=constr)

    def bind_sum(self, set_or_dict, constr=None):
        return self.bind("sum", set_or_dict=set_or_dict, constr=constr)

    def bind_ports(self, *ports):
        """
        TODO, do more type checking in here
        """
        ports = [tupleize(pt) for pt in ports]

        if len(ports) == 2:
            pB1 = self.pt[ports[0]]
            pB2 = self.pt[ports[1]]
            in1 = pB1.get("I", None)
            in2 = pB2.get("I", None)
            out1 = pB1.get("O", None)
            out2 = pB2.get("O", None)
            if in1 is not None and in2 is not None:
                raise NotImplementedError("TODO")
                return

        potentials = [self.pt[pt].potential for pt in ports]
        flows = [self.pt[pt].flow for pt in ports]

        self.bind_equal(potentials)
        self.bind_sum(flows)
        return

    def bind(self, form, set_or_dict, constr=None):
        """
        Binds multiple inputs together to be equal.
        if set_or_dict has only one element, it is bound to be zero
        """
        mdict = dict()
        klist = []
        if isinstance(set_or_dict, Mapping):
            for k, v in set_or_dict.items():
                k2 = tupleize(k)
                klist.append(k2)
                if v is 1:
                    v = None
                mdict[k2] = v
                assert k2 in self.io
        else:
            for k in set_or_dict:
                k2 = tupleize(k)
                klist.append(k2)
                mdict[k2] = None
                assert k2 in self.io

        # sign will alternate between pairs
        if len(klist) == 1:
            form = "zero"

        if form == "equal":
            # pre-setup the first of the two constraints
            io1 = klist[0]
            mul1 = mdict[io1]
            Cs1, N = self._Cs_collect(io1, mat=mul1)
            sign = 1
            for idx, io2 in enumerate(klist[1:]):
                sign = -1 * sign
                # todo, make nonzero constr able to handle multiple bindings
                if constr is None:
                    constr_use = tupleize((io1, io2))
                else:
                    if isinstance(constr, str):
                        assert len(klist) == 2
                        constr_use = [constr]
                    else:
                        assert isinstance(constr, list)
                    constr_use = tupleize(constr[idx])

                mul1 = mdict[io1]
                if mul1 is not None:
                    assert N == mul1.shape[-1]

                mul2 = mdict[io2]
                if mul2 is not None:
                    if sign == -1:
                        mul2 = -mul2
                    Cs2, N2 = self._Cs_collect(io2, mat=mul2)
                    assert N2 == N
                else:
                    Cs2, N2 = self._Cs_collect(io2, mat=sign)
                    assert N2 == N

                assert constr_use not in self.cr
                self.cr[constr_use].N = N

                for st, C in Cs1.items():
                    self.As[st, constr_use] = C
                    self.st2crA[st].add(constr_use)
                    self.cr2stA[constr_use].add(st)
                for st, C in Cs2.items():
                    self.As[st, constr_use] = C
                    self.st2crA[st].add(constr_use)
                    self.cr2stA[constr_use].add(st)

                io1 = io2
                Cs1 = Cs2
                mul1 = mul2
        elif form == "sum":
            if constr is None:
                constr_use = tupleize(klist)
            else:
                constr_use = tupleize(constr)
            assert constr_use not in self.cr

            N = None
            for idx, io in enumerate(klist):
                Cs, N2 = self._Cs_collect(io, mat=mdict[io])
                if N is None:
                    N = N2
                else:
                    assert N == N2

                for st, C in Cs.items():
                    self.As[st, constr_use] = C
                    self.st2crA[st].add(constr_use)
                    self.cr2stA[constr_use].add(st)
            self.cr[constr_use].N = N
        elif form == "zero":
            io1 = klist[0]
            mul1 = mdict[io1]
            Cs1, N = self._Cs_collect(io1, mat=mul1)
            # make a zero binding
            if constr is None:
                constr_use = ("zero", io1)
            else:
                constr_use = tupleize(constr)
            assert constr_use not in self.cr
            self.cr[constr_use].N = N
            for st, C in Cs1.items():
                self.As[st, constr_use] = C
                self.st2crA[st].add(constr_use)
                self.cr2stA[constr_use].add(st)
        else:
            raise RuntimeError("Binding type not recognized")

        return

    def annotate(self, annotation, io=None, st=None, cr=None):
        if io is not None:
            self.io[io].update(annotation)
        if st is not None:
            self.st[st].update(annotation)
        if cr is not None:
            self.cr[cr].update(annotation)
        return

    def check(self):
        # checks that there are enough A and E matrices for the intputs
        return

    def states_edges(self, st_ZR=None, include_cr=False):
        ststmap = defaultdict(set)
        if st_ZR is None:
            st_ZR = self.st2crA.keys()
        st_ZR = set(st_ZR)
        for st in st_ZR:
            cr_set = self.st2crA[st]
            for cr in cr_set:
                Erank1 = self.Eranks.get((st, cr), 0)
                if include_cr:
                    cr_s = cr
                for st2 in self.cr2stA[cr]:
                    if st2 not in st_ZR:
                        continue
                    Erank2 = self.Eranks.get((st2, cr), 0)
                    if include_cr:
                        ststmap[st, st2].add(
                            (
                                (st, cr) in self.As and (st2, cr) in self.As,
                                Erank1,
                                Erank2,
                                cr_s,
                            )
                        )
                    else:
                        ststmap[st, st2].add(
                            (
                                (st, cr) in self.As and (st2, cr) in self.As,
                                Erank1,
                                Erank2,
                            )
                        )
        return dict(ststmap)

    def debug_sparsity_print(self):
        printSSBnz(
            self.statespace([], [], Dreduce=False, allow_underconstrained=True),
        )

    def states_reducible(self):
        # purge the defaultdicts of null entries, so that
        # the set operations below succeed with the proper lists
        for st, cr_set in list(self.st2crE.items()):
            if not cr_set:
                self.st2crE.pop(st)
        for cr, st_set in list(self.cr2stE.items()):
            if not st_set:
                self.cr2stE.pop(cr)
        # states of unknown rank
        st_UR = set(self.st2crE)
        cr_UR = set(self.cr2stE)

        # states, constraints of zero rank
        st_ZR = set(self.st) - st_UR
        cr_ZR = set(self.cr) - cr_UR
        return (st_ZR, cr_ZR), (st_UR, cr_UR)

    # def _states_reducible_sccs_leafs(
    #        self,
    #        st_ZR,
    #        cr_ZR,
    #        st0s,
    #        cr0s,
    # ):
    #    st_collect = {st: {st, } for st in st_ZR}
    #    cr_collect = {cr: {cr, } for cr in cr_ZR}

    #    #this is a simple a fast algorithms for traversing down the leafs
    #    #currently just a backup to the Tarjan-based method below
    #    def reduce_leafs():
    #        nonlocal st0s
    #        nonlocal cr0s
    #        repeat = True
    #        while repeat:
    #            repeat = False
    #            if st0s:
    #                repeat = True
    #                st = st0s.pop()
    #                cr = next(iter(st2cr[st]))
    #                cr0s.discard(cr)
    #                sccs.append((st_collect[st], cr_collect[cr]))
    #                #now remove st and cr from others
    #                for st0 in cr2st[cr]:
    #                    cr_set = st2cr[st0]
    #                    cr_set.remove(cr)
    #                    if len(cr_set) == 1:
    #                        st0s.add(st0)
    #                del st2cr[st]
    #                del cr2st[cr]
    #                continue
    #            if cr0s:
    #                repeat = True
    #                cr = cr0s.pop()
    #                st = next(iter(cr2st[cr]))
    #                st0s.discard(st)
    #                sccs.append((st_collect[st], cr_collect[cr]))
    #                #now remove st and cr from others
    #                for cr0 in st2cr[st]:
    #                    st_set = cr2st[cr0]
    #                    st_set.remove(st)
    #                    if len(st_set) == 1:
    #                        cr0s.add(cr0)
    #                del st2cr[st]
    #                del cr2st[cr]
    #                continue
    #    reduce_leafs()

    def strongly_connected_components_reducible(self, st_start=None):
        """
        Goes through the reducible states and finds the strongly-connected components
        the set of these can be merged, and the resulting objects can be reduced

        This uses a modified Tarjan algorithm
        """
        (st_ZR, cr_ZR), _ = self.states_reducible()

        if True:

            def dprint(*args):
                return

        else:
            dprint = print

        # these are subset files of just the reducible states
        st2cr = dict()
        for st in st_ZR:
            st2cr[st] = self.st2crA[st] & cr_ZR
        cr2st = dict()
        for cr in cr_ZR:
            cr2st[cr] = self.cr2stA[cr] & st_ZR

        # this are tuples of ((states), (constr))
        sccs = []

        st_index = dict()
        cr_index = dict()
        st_lowlink = dict()
        cr_lowlink = dict()

        leafs_st = []
        leafs_cr = []
        # now find an scc, start by picking any state
        # this uses a modified Tarjan algorithm
        # stack alternates cr's and st's
        stack = []

        index = 0

        def recurse_st(st):
            nonlocal index
            nonlocal stack
            st_index[st] = index
            st_lowlink[st] = index
            index += 1
            my_index = len(stack)
            stack.append(st)
            dprint("rec st", st)

            cr_set = st2cr[st]
            while cr_set:
                cr = cr_set.pop()
                # must remove the back-link to prevent immediate re-covering
                cr2st[cr].remove(st)
                if cr not in cr_index:
                    dprint("new cr", cr)
                    recurse_cr(cr)
                    st_lowlink[st] = min(st_lowlink[st], cr_lowlink[cr])
                else:
                    dprint("old cr", cr)
                    st_lowlink[st] = min(st_lowlink[st], cr_index[cr])
            del st2cr[st]

            dprint("stack", stack)
            if st_lowlink[st] == st_index[st]:
                scc = stack[my_index:]
                stack = stack[:my_index]
                dprint("--------------------------scc_st", scc)
                if len(scc) == 0:
                    assert False
                elif len(scc) == 1:
                    leafs_st.extend(scc)
                else:
                    sccs.append((scc[0:-1:2], scc[1::2]))
                    dprint("scc:", sccs[-1])
            return

        def recurse_cr(cr):
            nonlocal index
            nonlocal stack
            cr_index[cr] = index
            cr_lowlink[cr] = index
            index += 1
            my_index = len(stack)
            stack.append(cr)
            dprint("rec cr", cr)

            st_set = cr2st[cr]
            while st_set:
                st = st_set.pop()
                # must remove the back-link to prevent immediate re-covering
                st2cr[st].remove(cr)
                if st not in st_index:
                    dprint("new st", st)
                    recurse_st(st)
                    cr_lowlink[cr] = min(cr_lowlink[cr], st_lowlink[st])
                else:
                    dprint("old st", st)
                    cr_lowlink[cr] = min(cr_lowlink[cr], st_index[st])
            del cr2st[cr]

            dprint("stack", stack)
            if cr_lowlink[cr] == cr_index[cr]:
                scc = stack[my_index:]
                stack = stack[:my_index]
                dprint("-------------------------scc_cr", scc)
                if len(scc) == 0:
                    assert False
                elif len(scc) == 1:
                    leafs_cr.extend(scc)
                else:
                    sccs.append((scc[1::2], scc[0:-1:2]))
                    dprint("scc:", sccs[-1])

        while st2cr:
            if st_start:
                st = tupleize(st_start)
                st_start = None
            else:
                st = next(iter(st2cr))
            recurse_st(st)

        # debugging test for size invariants
        for st_set, cr_set in sccs:
            Nst = sum(self.st[st].N for st in st_set)
            Ncr = sum(self.cr[cr].N for cr in cr_set)
            assert Nst == Ncr
        dprint(sccs, leafs_st, leafs_cr)
        return sccs, leafs_st, leafs_cr

    def simplify_scc(self, st_set, cr_set):
        """
        Reduces on an SCC
        """
        # TODO, do a collect-reduce on multi-component SCCs
        if len(st_set) != 1 or len(cr_set) != 1:
            raise NotImplementedError()
        else:
            st = st_set.pop()
            cr = cr_set.pop()
            st = tupleize(st)
            cr = tupleize(cr)
        assert self.Eranks.get((st, cr), 0) == 0
        assert self.st[st].N == self.cr[cr].N

        A = self.As[st, cr]
        # add the negative to all of the values
        # as it enters in similarly to all equations
        AnI = -np.linalg.inv(A)

        for st2 in self.cr2stA[cr]:
            if st2 == st:
                continue
            Apre = self.As[st2, cr]
            for cr2 in self.st2crA[st]:
                if cr2 == cr or st2 == st:
                    continue
                Apost = self.As[st, cr2]
                self.st2crA[st2].add(cr2)
                self.cr2stA[cr2].add(st2)
                Aprev = self.As.get((st2, cr2), None)
                if Aprev is None:
                    self.As[st2, cr2] = Apost @ AnI @ Apre
                else:
                    self.As[st2, cr2] = Aprev + Apost @ AnI @ Apre

            for io2 in self.st2io[st]:
                Cpost = self.Cs[st, io2]
                self.st2io[st2].add(io2)
                self.io2st[io2].add(st2)
                Cprev = self.Cs.get((st2, io2), None)
                if Cprev is None:
                    self.Cs[st2, io2] = Cpost @ AnI @ Apre
                else:
                    self.Cs[st2, io2] = Cprev + Cpost @ AnI @ Apre

        # now purge the state and constraint
        for st2 in self.cr2stA[cr]:
            if st2 == st:
                continue
            self.st2crA[st2].remove(cr)
            del self.As[st2, cr]
        self.cr2stA[cr].clear()
        self.cr2stA[cr].add(st)
        self.st2crA[st].add(cr)
        # add it back
        self.As[st, cr] = A
        # and register it as a potential diagonalized constraint
        self.AinvDcr[cr] = AnI

        for st2 in self.cr2stE[cr]:
            self.st2crE[st2].remove(cr)
            del self.Es[st2, cr]
            del self.Eranks[st2, cr]
        del self.cr2stE[cr]

        self.check_mats()
        return

    def check_mats(self):
        for (st, cr), A in self.As.items():
            assert st in self.cr2stA[cr]
            assert cr in self.st2crA[st]

        for (st, cr), E in self.Es.items():
            assert st in self.cr2stE[cr]
            assert cr in self.st2crE[st]

        for (st, io), C in self.Cs.items():
            assert st in self.io2st[io]
            assert io in self.st2io[st]

        for st in self.st:
            for cr in self.st2crA[st]:
                self.As[st, cr]
                assert st in self.cr2stA[cr]

            for cr in self.st2crE[st]:
                self.Es[st, cr]
                assert st in self.cr2stE[cr]

            for io in self.st2io[st]:
                self.Cs[st, io]
                assert st in self.io2st[io]

        for cr in self.cr:
            for st in self.cr2stA[cr]:
                self.As[st, cr]
                assert cr in self.st2crA[st]

            for st in self.cr2stE[cr]:
                self.Es[st, cr]
                assert cr in self.st2crA[st]
        return

    def reduce(self, states=None, constr=None):
        raise NotImplementedError()
        return

    def statespace(
        self,
        inputs,
        outputs,
        noises=(),
        states=None,
        constr=None,
        Ediag=True,
        Dreduce=True,
        allow_underconstrained=False,
    ):
        """
        Condense the internal representation and return a descriptor state space
        for it.

        the states and constr lists define an ordering to use. Otherwise, an
        ordering is attempted to make the resulting E matrix as diagonal as possible.
        It will output the ranges of E diagonal, full rank, unknown rank, zero rank

        if you specify states or constr, then E_diagonal==0, as it cannot
        organize the E matrix to ensure diagonals.

        TODO, should be able to output sparse matrices
        """

        if states is not None or constr is not None:
            raise NotImplementedError("Can't specify states or constraints order yet")

        (st_ZR, cr_ZR), (st_UR, cr_UR) = self.states_reducible()

        st_D = dict()
        cr_D = dict()
        if Dreduce:
            for cr in self.AinvDcr:
                if len(self.cr2stA[cr]) == 1:
                    st = next(iter(self.cr2stA[cr]))
                    st_D[st] = cr
                    cr_D[cr] = st
                    st_ZR.remove(st)
                    cr_ZR.remove(cr)
            # dprint('D states', st_D)

        # diagonals
        st_DG = []
        cr_DG = []
        st2crDG = dict()
        for st, cr_set in self.st2crE.items():
            if len(cr_set) == 1:
                # get item from set
                cr = next(iter(cr_set))
                Erank = self.Eranks[st, cr]
                if Erank is -1:
                    st_DG.append(st)
                    cr_DG.append(cr)
                    st2crDG[st] = cr
                    continue
            for cr in cr_set:
                # currently assumes that no 0 links are in there
                # so test the invariant
                Erank = self.Eranks[st, cr]
                assert Erank is not 0
        st_DG = set(st_DG)
        cr_DG = set(cr_DG)

        st_UK = st_UR - st_DG
        cr_UK = cr_UR - cr_DG

        states_DG = sorted(st_DG, key=ACEKeyTuple)
        # TODO, perform SCC topo ordering on UK terms
        states_UK = sorted(st_UK, key=ACEKeyTuple)
        states_ZR = sorted(st_ZR, key=ACEKeyTuple)
        states = states_DG + states_UK + states_ZR
        statesN = np.cumsum([0] + [self.st[st].N for st in states])
        Nstates = {
            st: (Ns, Ne) for st, Ns, Ne in zip(states, statesN[:-1], statesN[1:])
        }

        constr_DG = [st2crDG[st] for st in states_DG]
        # TODO, perform SCC topo ordering on UK terms
        constr_UK = sorted(cr_UK, key=ACEKeyTuple)
        constr_ZR = sorted(cr_ZR, key=ACEKeyTuple)
        constr = constr_DG + constr_UK + constr_ZR
        constrN = np.cumsum([0] + [self.cr[cr].N for cr in constr])
        Nconstr = {
            cr: (Ns, Ne) for cr, Ns, Ne in zip(constr, constrN[:-1], constrN[1:])
        }

        N_cr_DG = N_st_DG = sum(self.st[st].N for st in st_DG)
        N_st_UK = sum(self.st[st].N for st in st_UK)
        N_cr_UK = sum(self.cr[cr].N for cr in cr_UK)

        N_st_ZR = sum(self.st[st].N for st in st_ZR)
        N_cr_ZR = sum(self.cr[cr].N for cr in cr_ZR)

        N_st_tot = N_st_DG + N_st_UK + N_st_ZR
        N_cr_tot = N_cr_DG + N_cr_UK + N_cr_ZR

        if not allow_underconstrained and N_st_tot > N_cr_tot:
            raise RuntimeError(
                "Insufficient Constraints. Add more constraints or add more bound inputs"
            )
        if not allow_underconstrained and N_st_tot < N_cr_tot:
            raise RuntimeError("Excessive Constraints. Big Problemo")
        # need to add the input constraints to N_cr_tot

        # dprint('diagonal', N_st_DG, [(st, self.st[st].N) for st in st_DG])
        # dprint('unknown', N_st_UK, N_cr_UK)
        # dprint('zerorank', N_st_ZR, N_cr_ZR)

        # dprint(st_DG, st_UK, st_ZR)
        # dprint(cr_DG, cr_UK, cr_ZR)

        def common_shape_type(Dmat):
            shapes = []
            arrs = []
            for M in Dmat.values():
                shapes.append(M.shape[:-2])
                arrs.append(M.dtype)
            dtype = np.find_common_type([], arrs)
            shapes = broadcast_shapes(shapes)
            return shapes, dtype

        shape, dtype = common_shape_type(self.As)
        A = np.zeros(shape + (N_cr_tot, N_st_tot), dtype=dtype)
        shape, dtype = common_shape_type(self.Es)
        E = np.zeros(shape + (N_cr_tot, N_st_tot), dtype=dtype)

        outCs = dict()
        outCsD = dict()
        outN = [0]
        outputs = [tupleize(out) for out in outputs]
        for out in outputs:
            assert out in self.io
            Cs, N = self._Cs_collect(out)
            assert N > 0
            for st, subC in Cs.items():
                if st in st_D:
                    outCsD[st, out] = subC
                else:
                    outCs[st, out] = subC
            outN.append(N)
        outN = np.cumsum(outN)
        Nout = {out: (Ns, Ne) for out, Ns, Ne in zip(outputs, outN[:-1], outN[1:])}
        N_out_tot = outN[-1]

        inBs = dict()
        inBsD = dict()
        inN = [0]
        inputs = [tupleize(In) for In in inputs]
        for In in inputs:
            cr = self.bound_io[In]
            N = self.cr[cr].N
            if cr in cr_D:
                inBsD[In, cr] = np.eye(N)
            else:
                inBs[In, cr] = np.eye(N)
            inN.append(N)
        inN = np.cumsum(inN)
        Nin = {In: (Ns, Ne) for In, Ns, Ne in zip(inputs, inN[:-1], inN[1:])}
        N_in_tot = inN[-1]

        shape, dtype = common_shape_type(inBs)
        B = np.zeros(shape + (N_cr_tot, N_in_tot), dtype=dtype)

        shape, dtype = common_shape_type(outCs)
        C = np.zeros(shape + (N_out_tot, N_st_tot), dtype=dtype)

        D = np.zeros((N_out_tot, N_in_tot))

        for st in states:
            for cr in self.st2crA[st]:
                subA = self.As[st, cr]
                Ns1, Ns2 = Nstates[st]
                Nc1, Nc2 = Nconstr[cr]
                A[..., Nc1:Nc2, Ns1:Ns2] = subA

            for cr in self.st2crE[st]:
                subE = self.Es[st, cr]
                Ns1, Ns2 = Nstates[st]
                Nc1, Nc2 = Nconstr[cr]
                E[..., Nc1:Nc2, Ns1:Ns2] = subE

        for (st, out), subC in outCs.items():
            Ns1, Ns2 = Nstates[st]
            No1, No2 = Nout[out]
            C[..., No1:No2, Ns1:Ns2] = subC

        # TODO, this is wrong, needs to assign into the specific constraint associated with the input
        # this CAN be done using a pseudo-inverse with the C array, but it is easier to assign it to the constrain
        for (In, cr), subB in inBs.items():
            Nc1, Nc2 = Nconstr[cr]
            Ni1, Ni2 = Nin[In]
            B[..., Nc1:Nc2, Ni1:Ni2] = -subB

        # need to create the edge list for the outCsD
        st2ioOutCsD = defaultdict(set)
        for (st, io) in outCsD.keys():
            st2ioOutCsD[st].add(io)

        # construct B and D matrices from outCsD and inBsD
        for (In, cr), subB in inBsD.items():
            Ainv = self.AinvDcr[cr]
            # loop through other states touched by constraint
            Ni1, Ni2 = Nin[In]
            # subB is just id matrix
            # AiB = Ainv @ subB
            AiB = Ainv
            # this is why the state must be reduced to produce B
            st = cr_D[cr]

            # B first
            for cr1 in self.st2crA[st]:
                if cr1 == cr:
                    continue
                Apost = self.As[st, cr1]
                Nc1, Nc2 = Nconstr[cr1]
                B[..., Nc1:Nc2, Ni1:Ni2] -= Apost @ AiB

            for out1 in st2ioOutCsD[st]:
                Cpost = outCsD[st, out1]
                No1, No2 = Nout[out1]
                D[..., No1:No2, Ni1:Ni2] -= Cpost @ AiB

        if not Ediag:
            descriptor = True
        else:
            descriptor = N_st_DG != N_st_tot

        retB = Bunch(
            ABCDE=(A, B, C, D, E),
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
            O=Nout,
            I=Nin,
            Nstates=Nstates,
            Nconstr=Nconstr,
            Nin=Nin,
            Nout=Nout,
        )
        if Ediag:
            retB.E_diagonal = N_st_DG
            retB.E_unkownrank = N_st_UK
            retB.E_zerorank = N_st_ZR
            retB.N_states = N_st_tot
        if not descriptor:
            retB.ABCD = (A, B, C, D)
        return retB


def nz(M):
    return 1 * (M != 0)


def printSSBnz(ssb, cr_order=[], st_order=[]):
    c_str = []
    for (i1, i2), key in sorted([(v, k) for k, v in ssb.Nconstr.items()]):
        if i2 - i1 == 0:
            continue
        try:
            idx = cr_order.index(key)
            c_str.append(str(idx) + ": " + str(key))
        except ValueError:
            c_str.append(str(key))
        if i2 - i1 > 1:
            for _ in range(i2 - i1 - 2):
                c_str.append("┃")
            c_str.append("┗")
    c_str = "\n".join(c_str)

    s_str = []
    s_str_list = []
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for kidx, ((i1, i2), key) in enumerate(
        sorted([(v, k) for k, v in ssb.Nstates.items()])
    ):
        if i2 - i1 == 0:
            continue
        s_str_list.append(alpha[kidx] + ": " + str(key))
        try:
            idx = st_order.index(key)
            s_str.append(alpha[kidx] + str(idx))
        except ValueError:
            s_str.append(alpha[kidx] + " ")
        if i2 - i1 > 1:
            for _ in range(i2 - i1 - 2):
                s_str.append("━━")
            s_str.append("┓ ")
    s_str = "  " + "".join(s_str)

    Astr = np.array2string(nz(ssb.A), max_line_width=np.nan, threshold=100 ** 2)
    Estr = np.array2string(nz(ssb.E), max_line_width=np.nan, threshold=30 ** 2)
    Bstr = np.array2string(nz(ssb.B), max_line_width=np.nan, threshold=100 ** 2)
    Cstr = np.array2string(nz(ssb.C), max_line_width=np.nan, threshold=100 ** 2)
    Dstr = np.array2string(nz(ssb.D), max_line_width=np.nan, threshold=100 ** 2)
    print(" | ".join(s_str_list))
    ziplines(
        "\n" + c_str,
        s_str + "\n" + Astr + "\n\n" + Cstr,
        "\n" + Bstr + "\n\n" + Dstr,
        "\n" + Estr,
        delim=" | ",
    )


def ziplines(*args, delim=""):
    import itertools

    widths = []
    for arg in args:
        w = max(len(line) for line in arg.splitlines())
        widths.append(w)
    for al in itertools.zip_longest(*[arg.splitlines() for arg in args], fillvalue=""):
        line = []
        for a, w in zip(al, widths):
            line.append(a + " " * (w - len(a)))
        print(delim.join(line))
