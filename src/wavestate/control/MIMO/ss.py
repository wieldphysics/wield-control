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

from ..statespace.dense import xfer_algorithms
from ..statespace.dense import ss_algorithms
from ..statespace import ssprint

from . import mimo
from .. import siso


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
        inout=None,
        hermitian: bool = True,
        time_symm: bool = False,
        dt=None,
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
                    outputs[k] = inputs

        self.inputs = inputs
        self.outputs = outputs

        def reverse(d, length):
            lst = [set() for i in len(length)]
            for k, idx in d.items():
                lst[idx].add(k)
            return lst

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt

        self.rev_inputs = reverse(inputs, self.B.shape[-1])
        self.rev_outputs = reverse(inputs, self.C.shape[-2])

    @property
    def ABCDE(self):
        if self.E is None:
            E = np.eye(self.A.shape[-1])
        else:
            E = self.E
        return self.A, self.B, self.C, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.E

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

    def __getitem__(self, key):
        row, col = key

        def apply_map(group, length):
            if isinstance(row, slice):
                # make the klst just be using the slice
                klst = range(length)[row]
            elif isinstance(row, (list, tuple)):
                klst = []
                for k in row:
                    if isinstance(row, str):
                        k_i = self.outputs[k]
                    else:
                        k_i = k
                    klst.append(k_i)
            else:
                # will be a single index
                if isinstance(row, str):
                    klst = self.outputs[k]
                else:
                    klst = k

        r = apply_map(row, self.C.shape[-2])
        c = apply_map(col, self.B.shape[-1])
        if isinstance(r, list):
            assert(isinstance(c, list))

            def map_into(rev, lst):
                d = {}
                for idx, k in enumerate(lst):
                    kset = rev[lst]
                    for v in kset:
                        d[v] = idx
                return d

            inputs = map_into(rev = self.rev_inputs, lst=c)
            outputs = map_into(rev = self.rev_outputs, lst=r)

            return self.__class__(
                A=self.A,
                B=self.B[..., :, c],
                C=self.C[..., r, :],
                D=self.D[..., r, c],
                E=self.E,
                inputs=inputs,
                outputs=outputs,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
            )
        else:
            assert(not isinstance(c, list))
            # neither are lists, so return a SISO object
            return siso.SISOStateSpace(
                A=self.A,
                B=self.B[..., :, c],
                C=self.C[..., r, :],
                D=self.D[..., r, c],
                E=self.E,
                hermitian = self.hermitian,
                time_symm = self.time_symm,
                dt=self.dt,
            )

    def rename(self, renames, which='both'):
        """
        Rename inputs and outputs of the statespace

        which: can be inputs, outputs, or both (the default).

        """
        raise NotImplementedError("TODO")
        return

    def response(self, row=None, col=None, *, f=None, w=None, s=None):
        if row is not None:
            raise NotImplementedError()
        if col is not None:
            raise NotImplementedError()

        domain = None
        if f is not None:
            domain = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(domain is None)
            domain = 1j * np.asarray(w)
        if s is not None:
            assert(domain is None)
            domain = np.asarray(s)

        return xfer_algorithms.ss2response_mimo(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            s=domain,
            idx_in=0,
            idx_out=0,
        )

    def __mul__(self, other):
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
        elif isinstance(other, numbers.Number):
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

    def feedback(self, connection_list=None, gain=1, connection_dict=None):
        """
        Feedback linkage for a single statespace
        """

        fbD = np.zeros((self.D.shape[-1], self.B.shape[-2]))

        if connection_list is not None:
            for tup in connection_list:
                if len(tup) < 3:
                    iname, oname = tup
                    val = gain
                else:
                    iname, oname, val = tup
                iidx = self.inputs[iname]
                oidx = self.outputs[oname]
                fbD[iidx, oidx] = val

        if connection_dict is not None:
            for (iname, oname), v in connection_dict.items():
                iidx = self.inputs[iname]
                oidx = self.outputs[oname]
                fbD[iidx, oidx] = v

        clD = np.inv(np.eye(self.D.shape[-1]) - fbD @ self.D)

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
            dt=self.dt,
        )


def ss(
    *args,
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
    return MIMOStateSpace(
        A, B, C, D, E,
        hermitian=True,
        time_symm=False,
        dt=None,
    )



