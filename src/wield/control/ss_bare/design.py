#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023 California Institute of Technology.
# SPDX-FileCopyrightText: © 2023 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import numpy as np

from .ss import BareStateSpace


def replicateSS(ss, dim):
    """
    Take a raw statespace and replicate it "dim" number of times
    """
    # TODO preserve flags
    Nstates = ss.A.shape[-1]
    Nconstr = ss.A.shape[-2]
    Ninputs = ss.B.shape[-1]
    Noutputs = ss.C.shape[-2]

    constrN = Nconstr * dim
    statesN = Nstates * dim
    inputsN = Ninputs * dim
    outputN = Noutputs * dim

    A = np.zeros((constrN, statesN))
    E = np.zeros((constrN, statesN))
    B = np.zeros((constrN, inputsN))
    C = np.zeros((outputN, statesN))
    D = np.zeros((outputN, inputsN))

    if E is not None:
        assert(E.shape == A.shape)

    for idx in range(dim):
        slc_s = slice(idx*Nstates, (idx+1)*Nstates)
        slc_c = slice(idx*Nconstr, (idx+1)*Nconstr)
        slc_i = slice(idx*Ninputs, (idx+1)*Ninputs)
        slc_o = slice(idx*Noutputs, (idx+1)*Noutputs)

        A[..., slc_c, slc_s] = ss.A
        E[..., slc_c, slc_s] = ss.E
        B[..., slc_c, slc_i] = ss.B[..., :, :]
        C[..., slc_o, slc_s] = ss.C[..., :, :]
        D[..., slc_o, slc_i] = ss.D[..., :, :]

    return BareStateSpace(
        A, B, C, D, E,
        hermitian=ss.hermitian,
        time_symm=ss.time_symm,
        dt=ss.dt,
    )
