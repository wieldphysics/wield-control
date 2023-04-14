#!/USSR/bin/env python

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
from collections.abc import Mapping
import numbers
import numpy as np
from copy import deepcopy

from wield.bunch import Bunch

from ..statespace.ss import RawStateSpace, RawStateSpaceUser

from . import mimo
from . import response
from . import util
from .. import SISO


class MIMOStateSpace(RawStateSpaceUser, mimo.MIMO):
    """
    State space class to represent MIMO Transfer functions using dense matrix representations

    This class allows both string-based and number based indexing

    inputs and outputs can either be a list of names or a dictionary of names to indices.

    inputs and outputs can contain overlapping indices
    """
    def __init__(
        self,
        ss,
        inputs=None,
        outputs=None,
        warn=True,
    ):
        # TODO, secondaries should perhaps only be done using a secondary call, not here
        inputs, in_secondaries = util.io_normalize(inputs, ss.Ninputs)
        outputs, out_secondaries = util.io_normalize(outputs, ss.Noutputs)

        if in_secondaries:
            raise NotImplementedError()

        if out_secondaries:
            raise NotImplementedError()

        rlst, outputs, olst, rlistified = util.apply_io_map(list(outputs), outputs)
        clst, inputs, ilst, clistified = util.apply_io_map(list(inputs), inputs)

        ilength = len(clst)
        idissect = Bunch(
            name='inputs',
            length=ilength,
            idx_start=0,
            idx_end=ilength,
            channels=ilst,
        )

        olength = len(rlst)
        odissect = Bunch(
            name='outputs',
            length=olength,
            idx_start=0,
            idx_end=olength,
            channels=olst,
        )

        # go ahead and reduce and reorder the statespace
        ss2 = ss[rlst, clst]
        output_dissections = [odissect]
        input_dissections = [idissect]

        self.inputs_rev = util.reverse_io_map(
            inputs,
            ss.Ninputs,
            "inputs",
            warn=warn
        )
        self.outputs_rev = util.reverse_io_map(
            outputs,
            ss.Noutputs,
            "outputs",
            warn=warn
        )

        self.inputs = inputs
        self.outputs = outputs
        super().__init__(ss=ss2)

        self.input_dissections = input_dissections
        self.output_dissections = output_dissections
        return

    @classmethod
    def __build__(
            cls,
            ss=None,
            inputs=None,
            outputs=None,
            input_dissections=None,
            output_dissections=None
    ):
        """
        The raw constructor. Performs minimal computations and few consistency checks

        input_dissections and output_dissections must be a list of Bunches. Each bunch must include
        dB.name : name of the dissection
        dB.length : length of channels
        dB.idx_start : start index of the inputs/outputs
        dB.idx_end : end index of the inputs/outputs (must be idx_start + length)
        dB.channel_list = [] # a list of channels in the dissection

        This constructor assumes that the dissections are complete
        """
        self = cls.__new__(cls)
        super(MIMOStateSpace, self).__init__(ss=ss)

        self.inputs = inputs
        self.outputs = outputs

        self.inputs_rev = util.reverse_io_map(
            inputs,
            ss.Ninputs,
            "inputs",
            warn=False,
        )
        self.outputs_rev = util.reverse_io_map(
            outputs,
            ss.Noutputs,
            "outputs",
            warn=False,
        )

        self.input_dissections = input_dissections
        self.output_dissections = output_dissections
        return self

    def secondaries(inputs=None, outputs=None):
        """
        Return a new system with additional inputs and outputs created from the names of the originals

        Properly implementing should use a topological sort on inputs and outputs to check for cycles
        and ensure that the index dependency is resolvable
        """
        raise NotImplementedError()

    def siso(self, row, col):
        """
        convert a single output (row) and input (col) into a SISO
        representation
        """
        r = self.outputs[row]
        if isinstance(r, tuple):
            raise RuntimeError("Row name is a span and cannot be used to create a SISO system")
        c = self.inputs[col]
        if isinstance(c, tuple):
            raise RuntimeError("Col name is a span and cannot be used to create a SISO system")
        return SISO.SISOStateSpace(
            self.ss[r:r+1, c:c+1],
        )

    def __getitem__(self, key):
        """
        key should be a
        ss[[output_list_row, ...], [input_list_col, ...]] which will return another MIMO object

        ss[output_chn, input_chn] which will return a SISO object losing channel information

        drops dissection information if it existed
        """
        row, col = key

        rlst, outputs, olst, rlistified = util.apply_io_map(row, self.outputs)
        clst, inputs, ilst, clistified = util.apply_io_map(col, self.inputs)

        ilength = len(clst)
        idissect = Bunch(
            name='inputs',
            length=ilength,
            idx_start=0,
            idx_end=ilength,
            channels=ilst,
        )

        olength = len(rlst)
        odissect = Bunch(
            name='outputs',
            length=olength,
            idx_start=0,
            idx_end=olength,
            channels=olst,
        )

        if rlistified:
            if not clistified:
                raise RuntimeError("Both the row and col must either be given as a single element (SISO), or as a collection (MIMO)")
            r, = rlst
            c, = clst
            return SISO.SISOStateSpace(
                self.ss[r:r+1, c:c+1],
            )
        else:
            if clistified:
                raise RuntimeError("Both the row and col must either be given as a single element (SISO), or as a collection (MIMO)")
            return self.__build__(
                ss=self.ss[rlst, clst],
                inputs=inputs,
                outputs=outputs,
                output_dissections=[odissect],
                input_dissections=[idissect],
            )

    def dissect(
            self,
            *,
            ilists,
            inames,
            olists,
            onames,
    ):
        """
        This implements the dissection interface to break the statespace into
        blocks. This is useful for a number of advanced control synthesis and
        analysis routines that depend on categories of input and output blocks
        """

        ichannels = []
        iset = set()
        for ilst in ilists:
            assert(iset.isdisjoint(ilst))
            iset.update(ilst)

            ichannels.extend(ilst)

        ochannels = []
        oset = set()
        for olst in olists:
            assert(oset.isdisjoint(olst))
            oset.update(olst)

            ochannels.extend(olst)

        rlst, outputs, olst, rlistified = util.apply_io_map(ochannels, self.outputs)
        clst, inputs, ilst, clistified = util.apply_io_map(ochannels, self.inputs)
        ss = self.ss[rlst, clst]

        input_dissections = []
        for ilst, iname in zip(ilists, inames):
            idx_st = inputs[ilst[0]]
            if isinstance(idx_st, tuple):
                idx_st = idx_st[0]
            idx_ed = inputs[ilst[-1]]
            if isinstance(idx_ed, tuple):
                idx_ed = idx_st[1]
            else:
                idx_ed += 1
            ilength = idx_ed - idx_st
            idissect = Bunch(
                name=iname,
                length=ilength,
                idx_start=idx_st,
                idx_end=idx_ed,
                channels=ilst,
            )
            input_dissections.append(idissect)

        output_dissections = []
        for olst, oname in zip(olists, onames):
            idx_st = outputs[olst[0]]
            if isinstance(idx_st, tuple):
                idx_st = idx_st[0]
            idx_ed = outputs[ilst[-1]]
            if isinstance(idx_ed, tuple):
                idx_ed = idx_st[1]
            else:
                idx_ed += 1
            olength = idx_ed - idx_st
            odissect = Bunch(
                name=oname,
                length=olength,
                idx_start=idx_st,
                idx_end=idx_ed,
                channels=olst,
            )
            output_dissections.append(odissect)

        return self.__build__(
            ss=ss,
            inputs=inputs,
            outputs=outputs,
            output_dissections=output_dissections,
            input_dissections=input_dissections,
        )

    def namespace(self, ns):
        """
        prepend a namespace to all inputs and outputs and return the new system
        """
        inputs2 = {ns + k: v for k, v in self.inputs.items()}
        outputs2 = {ns + k: v for k, v in self.outputs.items()}

        input_dissections = []
        for idB in self.input_dissections:
            idB2 = deepcopy(idB)
            idB2.channels = [ns + k for k in idB.channels]
            input_dissections.append(idB2)

        output_dissections = []
        for odB in self.input_dissections:
            odB2 = deepcopy(odB)
            odB2.channels = [ns + k for k in odB.channels]
            output_dissections.append(odB2)

        return self.__build__(
            ss=self.ss,
            inputs=inputs2,
            outputs=outputs2,
            input_dissections=input_dissections,
            output_dissections=output_dissections,
        )

    def rename(self, renames, which='both'):
        """
        Rename inputs and outputs of the statespace and return the new systems

        renames: dictionary mapping from:to name pairs or a function(from) -> to
        which: can be "inputs", "outputs", or "both" (the default).
        """
        assert(which in ['both', 'inputs', 'outputs'])
        if isinstance(renames, Mapping):
            inputs2 = dict(self.inputs)
            if which == 'both' or which == 'inputs':
                for k, v in renames.items():
                    inputs2[v] = inputs2[k]
                    del inputs2[k]
                input_dissections = []
                for idB in self.input_dissections:
                    idB2 = deepcopy(idB)
                    idB2.channels = [renames.get(k, k) for k in idB.channels]
                    input_dissections.append(idB2)
            else:
                input_dissections = deepcopy(self.input_dissections)

            outputs2 = dict(self.outputs)
            if which == 'both' or which == 'outputs':
                for k, v in renames.items():
                    outputs2[v] = outputs2[k]
                    del outputs2[k]
                output_dissections = []
                for odB in self.input_dissections:
                    odB2 = deepcopy(odB)
                    odB2.channels = [renames.get(k, k) for k in odB.channels]
                    output_dissections.append(odB2)
            else:
                output_dissections = deepcopy(self.output_dissections)


        elif callable(renames):
            inputs2 = dict()
            if which == 'both' or which == 'inputs':
                for k, v in self.inputs.items():
                    inputs2[renames(k)] = v
                input_dissections = []
                for idB in self.input_dissections:
                    idB2 = deepcopy(idB)
                    idB2.channels = [renames(k) for k in idB.channels]
                    input_dissections.append(idB2)
            else:
                inputs2 = self.inputs
                input_dissections = deepcopy(self.input_dissections)

            outputs2 = dict()
            if which == 'both' or which == 'outputs':
                for k, v in self.outputs.items():
                    outputs2[renames(k)] = v
                output_dissections = []
                for odB in self.input_dissections:
                    odB2 = deepcopy(odB)
                    odB2.channels = [renames(k) for k in odB.channels]
                    output_dissections.append(odB2)
            else:
                outputs2 = self.outputs
                output_dissections = deepcopy(self.output_dissections)
        else:
            raise RuntimeError("Don't recognize renames type. Should be either a mapping or a callable")

        return self.__build__(
            ss=self.ss,
            inputs=inputs2,
            outputs=outputs2,
            input_dissections=input_dissections,
            output_dissections=output_dissections,
        )

    def rename_inputs(self, renames):
        return self.rename(renames=renames, which='inputs')

    def rename_outputs(self, renames):
        return self.rename(renames=renames, which='outputs')

    def fresponse(self, *, f=None, w=None, s=None):
        tf = self.ss.fresponse_raw(f=f, s=s, w=w)
        return response.MIMOFResponse(
            tf=tf,
            w=w,
            f=f,
            s=s,
            inputs=self.inputs,
            outputs=self.outputs,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            snr=None,
        )

    def in2out(self, inputs=None):
        raise NotImplementedError()

    def out2in(self, outputs=None):
        raise NotImplementedError()

    def inverse(self, inputs, outputs):
        """
        Creates the inverse between the set of inputs and outputs.
        the size of inputs and outputs must be the same.
        """
        raise NotImplementedError()

    def constraint(self, outputs=None, matrix=None):
        """
        Adds an output constraint to the system and returns the altered system

        outputs: this is a list of outputs which establishes an order
        matrix: this is a matrix for the list of outputs which adds the system constraint
        G:=matrix -> G @ C @ x = 0 by augmenting the A and E matrices
        """
        raise NotImplementedError()

    def constraints(self, output_matrix=[]):
        """
        Adds multiple output constraints to the system and returns the altered system

        output_matrix is a list of output, matrix pairs. This function is
        equivalent to calling constraint many times with the list, but is faster
        to perform all at once
        """
        raise NotImplementedError()

    def series_connect(self, *, input_connections=None, output_connections=None, gain=1):
        """
        Like feedback but extends inputs and outputs through gain or gain systems in series

        the gain terms can be SISO systems, StateSpaceRaw, D, ABCD, or ABCDE blocks
        """
        raise NotImplementedError()

    def feedback_connect(self, *, connections=None, gain=1):
        """
        Feedback linkage for a single statespace.

        connections is a list of row, col pairs or row,col,gain tuples

        gain is the connection gain to apply. It can be a scalar or a matrix

        TODO: allow gain to be a SISO response, StateSpaceRaw, D, ABCD, or ABCDE blocks
        """
        fbD = np.zeros((self.D.shape[-1], self.D.shape[-2]))

        if isinstance(connections, (list, tuple, set)):
            for tup in connections:
                if len(tup) < 3:
                    iname, oname = tup
                    val = gain
                else:
                    iname, oname, val = tup
                cidx = self.inputs[iname]
                ridx = self.outputs[oname]
                # note that the usual row and col conventions
                # are reversed in fbB since it is a feedback matrix
                is_matrix = False
                if isinstance(cidx, tuple):
                    is_matrix = True
                if isinstance(ridx, tuple):
                    is_matrix = True
                if is_matrix:
                    # promote to span if not already
                    if not isinstance(cidx, tuple):
                        cidx = (cidx, cidx+1)
                    if not isinstance(ridx, tuple):
                        ridx = (ridx, ridx+1)
                    cidxA, cidxB = cidx
                    ridxA, ridxB = ridx
                    if isinstance(val, numbers.Number):
                        # must be the same size if using a scalar gain!
                        assert(cidxB - cidxA == ridxB - ridxA)
                        fbD[..., cidxA:cidxB, ridxA:ridxB] = np.eye(cidxB - cidxA) * val
                    else:
                        # assuming val is a matrix gain
                        fbD[..., cidxA:cidxB, ridxA:ridxB] = val
                else:
                    fbD[..., cidx, ridx] = val

        elif isinstance(connections, Mapping):
            for (iname, oname), v in connections.items():
                iidx = self.inputs[iname]
                oidx = self.outputs[oname]
                if v is None:
                    v = gain
                fbD[iidx, oidx] = v

        ss = self.ss.feedbackD(D=fbD)

        return self.__build__(
            ss=ss,
            inputs=deepcopy(self.inputs),
            outputs=deepcopy(self.outputs),
            input_dissections=deepcopy(self.input_dissections),
            output_dissections=deepcopy(self.output_dissections),
        )


def statespace(
    *args,
    inputs=None,
    outputs=None,
    inout=None,
    hermitian=True,
    time_symm=False,
    dt=None,
):
    """
    Form a MIMO LTI system from statespace matrices.

    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, MIMOStateSpace):
            return arg
        elif isinstance(arg, (tuple, list)):
            if len(arg) == 4:
                A, B, C, D = arg
                E = None
            elif len(arg) == 5:
                A, B, C, D, E = arg
            else:
                raise RuntimeError("Unrecognized argument format")
    elif len(args) == 4:
        A, B, C, D = args
        E = None
    elif len(args) == 5:
        A, B, C, D, E = args
    else:
        raise RuntimeError("Unrecognized argument format")

    if inputs is not None:
        if isinstance(inputs, (list, tuple)):
            # convert to a dictionary
            inputs = {k: i for i, k in enumerate(inputs)}
    else:
        inputs = {}

    if outputs is not None:
        if isinstance(outputs, (list, tuple)):
            # convert to a dictionary
            outputs = {k: i for i, k in enumerate(outputs)}
    else:
        outputs = {}

    if inout is not None:
        for k, idx in inout.items():
            if k.endswith('.in'):
                is_output = False
                k = k[:-3]
            elif k.endswith('.i'):
                is_output = False
                k = k[:-2]
            elif k.endswith('.out'):
                is_output = True
                k = k[:-4]
            elif k.endswith('.o'):
                is_output = True
                k = k[:-2]
            else:
                raise RuntimeError("inout dict has key {} which does not end with .in, .i, .out, or .o".format(k))

            if is_output:
                assert(k not in outputs)
                outputs[k] = idx
            else:
                assert(k not in inputs)
                inputs[k] = idx

    return MIMOStateSpace(
        ss=RawStateSpace(
            A, B, C, D, E,
            hermitian=hermitian,
            time_symm=time_symm,
            dt=dt,
        ),
        inputs=inputs,
        outputs=outputs,
    )


def ssjoinsum(*args):
    """
    Join a list of MIMO state spaces into a single larger space. Common inputs
    will be connected and common outputs will be summed.
    """
    # TODO preserve flags
    SSs = args
    inputs = {}
    outputs = {}

    def aggregate(local_d, outer_d, outerN):
        outerNagg = outerN
        for name, key in local_d.items():
            if isinstance(key, tuple):
                st, sp = key
                prev = outer_d.get(name, None)
                if prev is not None:
                    pst, psp = prev
                    assert(psp - pst == sp - st)
                else:
                    outer_d[name] = (st + outerN, sp + outerN)
                    outerNagg += sp - st
            else:
                prev = outer_d.get(name, None)
                if prev is not None:
                    pass
                else:
                    outer_d[name] = key + outerN
                    outerNagg += 1
        return outerNagg

    ss_seq = []
    constrN = 0
    statesN = 0
    inputsN = 0
    outputN = 0
    for idx, ss in enumerate(SSs):
        ssB = Bunch()
        A, B, C, D, E = ss.ABCDE
        ssB.A = A
        ssB.B = B
        ssB.C = C
        ssB.D = D
        ssB.E = E
        ssB.inputs = ss.inputs
        ssB.outputs = ss.outputs
        ssB.sN = slice(statesN, statesN + A.shape[-2])
        ssB.cN = slice(constrN, constrN + A.shape[-1])
        if E is not None:
            assert(E.shape == A.shape)

        constrN += A.shape[-2]
        statesN += A.shape[-1]
        ss_seq.append(ssB)

        inputsN = aggregate(ss.inputs, inputs, inputsN)
        outputN = aggregate(ss.outputs, outputs, outputN)
        if idx == 0:
            dt = ss.dt
        else:
            assert(ss.dt == dt)

    A = np.zeros((constrN, statesN))
    E = np.zeros((constrN, statesN))
    B = np.zeros((constrN, inputsN))
    C = np.zeros((outputN, statesN))
    D = np.zeros((outputN, inputsN))

    for idx_ss, ssB in enumerate(ss_seq):
        A[..., ssB.cN, ssB.sN] = ssB.A
        E[..., ssB.cN, ssB.sN] = ssB.E

        def toslc(key_to, key_fr):
            if isinstance(key_fr, tuple):
                slc_fr = slice(key_fr[0], key_fr[1])
                slc_to = slice(key_to[0], key_to[1])
            else:
                slc_fr = key_fr
                slc_to = key_to
            return slc_to, slc_fr

        # TODO, this is probably slow and could be sped up using
        # some pre-blocking in the aggregate function above
        for name, key_fr in ssB.inputs.items():
            key_to = inputs[name]
            islc_to, islc_fr = toslc(key_to, key_fr)
            B[..., ssB.cN, islc_to] = ssB.B[..., :, islc_fr]

        for name, key_fr in ssB.outputs.items():
            key_to = outputs[name]
            oslc_to, oslc_fr = toslc(key_to, key_fr)
            C[..., oslc_to, ssB.sN] = ssB.C[..., oslc_fr, :]

            for name, key_fr in ssB.inputs.items():
                key_to = inputs[name]
                islc_to, islc_fr = toslc(key_to, key_fr)
                D[..., oslc_to, islc_to] = ssB.D[..., oslc_fr, islc_fr]

    return MIMOStateSpace(
        ss=RawStateSpace(
            A, B, C, D, E,
            hermitian=np.all(ss.hermitian for ss in SSs),
            time_symm=np.all(ss.time_symm for ss in SSs),
            dt=SSs[0].dt,
        ),
        inputs=inputs,
        outputs=outputs,
    )

