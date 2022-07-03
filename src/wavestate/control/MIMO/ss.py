#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numbers
import numpy as np
import warnings

from wavestate.bunch import Bunch

from ..statespace.dense import xfer_algorithms
from ..statespace.dense import ss_algorithms
from ..statespace import ssprint

from . import mimo
from . import response
from .. import SISO


class MIMOStateSpace(mimo.MIMO):
    """
    State space class to represent MIMO Transfer functions using dense matrix representations

    This class allows both string-based and number based indexing

    inputs and outputs can either be a list of names or a dictionary of names to indices.

    inout can be specified instead of inputs and outputs and it must be a dictionary with both
    inputs and outputs. If specified, each key must contain ".in" or ".out" as the last characters.
    """
    def __init__(
        self,
        A, B, C, D, E,
        inputs=None,
        outputs=None,
        hermitian: bool = True,
        time_symm: bool = False,
        flags={},
        dt=None,
        warn=True,
    ):
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        if E is not None:
            E = np.asarray(E)

        if hermitian:
            assert(np.all(A.imag == 0))
            assert(np.all(B.imag == 0))
            assert(np.all(C.imag == 0))
            assert(np.all(D.imag == 0))
            if E is not None:

                assert(np.all(E.imag == 0))

        def idx_normalize(idx):
            if isinstance(idx, (tuple, list)):
                st, sp = idx
                return (st, sp)
            if isinstance(idx, slice):
                assert(idx.span is None)
                return (idx.start, idx.stop)
            return idx

        if inputs is not None:
            if isinstance(inputs, (list, tuple)):
                # convert to a dictionary
                inputs = {k: i for i, k in enumerate(inputs)}
            else:
                # normalize
                inputs = {k: idx_normalize(v) for k, v in inputs.items()}
        else:
            inputs = {}

        if outputs is not None:
            if isinstance(outputs, (list, tuple)):
                # convert to a dictionary
                outputs = {k: i for i, k in enumerate(outputs)}
            else:
                # normalize
                outputs = {k: idx_normalize(v) for k, v in outputs.items()}
        else:
            outputs = {}

        self.flags = flags
        self.inputs = inputs
        self.outputs = outputs

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt

        def reverse(d, length, io):
            rev = {}
            lst = np.zeros(length, dtype=bool)
            for k, idx in d.items():
                if isinstance(idx, tuple):
                    st, sp = idx
                else:
                    prev = rev.setdefault(idx, k)
                    if lst[idx]:
                        raise RuntimeError("Overlapping indices")
                    lst[idx] = True
            if warn and not np.all(lst):
                warnings.warn("state space has under specified {}".format(io))
            return rev

        self.inputs_rev = reverse(inputs, self.B.shape[-1], "inputs")
        self.outputs_rev = reverse(outputs, self.C.shape[-2], "outputs")
        return

    @property
    def ABCDE(self):
        if self.E is None:
            E = np.eye(self.A.shape[-1])
        else:
            E = self.E
        return self.A, self.B, self.C, self.D, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.D, self.E

    @property
    def ABCD(self):
        if self.E is None:
            raise RuntimeError("Cannot Drop E")
        else:
            assert(np.all(np.eye(self.E.shape[-1]) == self.E))
            self.E = None
        return self.A, self.B, self.C, self.D

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        yield self.A
        yield self.B
        yield self.C
        yield self.D
        if self.E is not None:
            yield self.E

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self)

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
            raise RuntimeError("Row name is a span and cannot be used to create a SISO system")
        ret = SISO.SISOStateSpace(
            A=self.A,
            B=self.B[..., :, c:c+1],
            C=self.C[..., r:r+1, :],
            D=self.D[..., r:r+1, c:c+1],
            E=self.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )

        print("SHAPE2", self.D.shape, ret.D.shape)
        return ret

    def __getitem__(self, key):
        row, col = key

        def apply_map(group, length, dmap):
            d = {}
            if isinstance(group, slice):
                raise RuntimeError("Slices are not supported on MIMOStateSpace")
            elif isinstance(group, (list, tuple, set)):
                pass
            else:
                # normalize to use a list
                group = [group]

            klst = []
            for k in group:
                if isinstance(k, str):
                    idx = dmap[k]
                    if isinstance(idx, tuple):
                        st = len(klst)
                        klst.extend(range(idx[0], idx[1]))
                        sp = len(klst)
                        d[k] = (st, sp)
                    else:
                        d[k] = len(klst)
                        klst.append(idx)
                else:
                    name = self.inputs_rev[k]
                    d[name] = len(klst)
                    klst.append(k)
            return klst, d

        r, outputs = apply_map(row, self.C.shape[-2], self.outputs)
        c, inputs = apply_map(col, self.B.shape[-1], self.inputs)

        ret = self.__class__(
            A=self.A,
            B=self.B[..., :, c],
            C=self.C[..., r, :],
            D=self.D[..., r, :][..., :, c],  # annoying way that multiple list indices are grouped by numpy
            E=self.E,
            inputs=inputs,
            outputs=outputs,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )
        print("SHAPE", self.D.shape, ret.D.shape)
        return ret

    def rename(self, renames, which='both'):
        """
        Rename inputs and outputs of the statespace

        which: can be inputs, outputs, or both (the default).

        """
        raise NotImplementedError("TODO")
        return

    def response(self, *, f=None, w=None, s=None):
        domain = None
        if f is not None:
            domain = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(domain is None)
            domain = 1j * np.asarray(w)
        if s is not None:
            assert(domain is None)
            domain = np.asarray(s)

        tf = xfer_algorithms.ss2response_mimo(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            s=domain,
            idx_in=0,
            idx_out=0,
        )
        return response.MIMOResponse(
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

    def __matmul__(self, other):
        """
        """
        if isinstance(other, mimo.MIMOStateSpace):
            # currently need to do some checking about the inputs
            # and the outputs
            return NotImplemented
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            assert(self.dt == other.dt)
            ABCDE = ss_algorithms.chain([self.ABCDE, other.ABCDE])
            return self.__class__(
                A=ABCDE.A,
                B=ABCDE.B,
                C=ABCDE.C,
                D=ABCDE.D,
                E=ABCDE.E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B * other,
                C=self.C,
                D=self.D * other,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=other * self.C,
                D=other * self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            return NotImplemented

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
        Adds an output constraint to the system.

        outputs: this is a list of outputs which establishes an order
        matrix: this is a matrix for the list of outputs which adds the system constraint
        G:=matrix -> G @ C @ x = 0 by augmenting the A and E matrices
        """
        raise NotImplementedError()

    def constraints(self, output_matrix=[]):
        """
        Adds multiple output constraints to the system.

        output_matrix is a list of output, matrix pairs. This function is
        equivalent to calling constraint many times with the list, but is faster
        to perform all at once
        """
        raise NotImplementedError()

    def feedback(self, connections=None, gain=1):
        """
        Feedback linkage for a single statespace

        connections_rowcol is a list of row, col pairs
        gain is the connection gain to apply
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
                if isinstance(cidx, tuple):
                    assert(isinstance(ridx, tuple))
                    cidxA, cidxB = cidx
                    ridxA, ridxB = ridx
                    assert(cidxB - cidxA == ridxB - ridxA)
                    fbD[..., cidxA:cidxB, ridxA:ridxB] = np.eye(cidxB - cidxA) * val
                else:
                    fbD[..., cidx, ridx] = val

        elif isinstance(connections, dict):
            for (iname, oname), v in connections.items():
                iidx = self.inputs[iname]
                oidx = self.outputs[oname]
                if v is None:
                    v = gain
                fbD[iidx, oidx] = v

        # TODO, could prepare an LU decomposition for this
        clD = np.linalg.solve(np.eye(self.D.shape[-1]) - fbD @ self.D, fbD)

        if self.dt is not None:
            raise NotImplementedError("feedback not yet implemented in discrete time")

        A = self.A + self.B @ clD @ self.C
        B = self.B + self.B @ clD @ self.D
        C = self.C + self.D @ clD @ self.C
        D = self.D + self.D @ clD @ self.D

        return self.__class__(
            A,
            B,
            C,
            D,
            self.E,
            inputs=self.inputs,
            outputs=self.outputs,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
        )


def ss(
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
            A, B, C, D, E = arg
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
        A, B, C, D, E,
        inputs=inputs,
        outputs=outputs,
        hermitian=hermitian,
        time_symm=time_symm,
        dt=dt,
    )


def ssjoinsum(*args):
    """
    Join a list of MIMO state spaces into a single larger space. Common inputs
    will be connected and common outputs will be summed.
    """
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
        A, B, C, D, E,
        inputs=inputs,
        outputs=outputs,
        hermitian=np.all(ss.hermitian for ss in SSs),
        time_symm=np.all(ss.time_symm for ss in SSs),
        dt=SSs[0].dt,
    )

