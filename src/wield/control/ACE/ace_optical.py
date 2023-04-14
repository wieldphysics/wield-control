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
from wield import declarative
from collections import defaultdict, Mapping
import functools
from numbers import Number

from wield.control.algorithms.statespace import dense
from wield.control.statespace import ACE
from wield.control.algorithms.statespace.dense.zpk_algorithms import zpk_cascade
from wield.control.algorithms.statespace.dense.xfer_algorithms import ss2xfer

from wield.quantum import qop

from wield.bunch import FrozenBunch

c_m_s = 299792458


class ACEOpticalSetup(object):
    def __init__(self, pol=False, modes="HG01"):
        self.fname2fHz = dict()
        self.idx2fHz = []
        self.idx2fname = []
        self.linear_phase_band_Hz = 1e4

        if pol:
            self.basis_pol = FrozenBunch(
                type="pol",
                basis="PS",
                N=2,
            )
        else:
            self.basis_pol = FrozenBunch(
                type="pol",
                basis="None",
                N=1,
            )

        # this is the basis for the quadratures
        self.basis_qp = FrozenBunch(
            type="quadratures",
            basis="amp/phase:Q/P",
            N=2,
        )

        # DC code is currently only for plane-wave/HG00 solution
        self.basis_modesDC = FrozenBunch(
            type="modes",
            basis="HG:00",
            N=1,
        )

        if modes == "HG01":
            self.basis_modesAC = FrozenBunch(
                type="modes",
                basis="HG:00,10,01",
                N=2,
            )
        elif modes == "HG00":
            self.basis_modesAC = FrozenBunch(
                type="modes",
                basis="HG:00",
                N=1,
            )
        else:
            raise RuntimeError("Unrecognized modes")

        self.Ninternal = self.basis_qp.N * self.basis_pol.N * self.basis_modesAC.eye

        self.qAE_ExpandDC = qop(
            ("qp", "pol", "modes"),
            (self.basis_qp, self.basis_pol, self.basis_modesDC),
            (self.basis_qp, self.basis_pol, self.basis_modesDC),
            np.eye(self.Ninternal),
        )
        self.qAE_ExpandAC = qop(
            ("qp", "pol", "modes"),
            (self.basis_qp, self.basis_pol, self.basis_modesAC),
            (self.basis_qp, self.basis_pol, self.basis_modesAC),
            np.eye(self.Ninternal),
        )
        # these don't expand in the input or outputs states
        self.qBC_ExpandDC = qop(
            ("qp", "pol", "modes"),
            (self.basis_qp, self.basis_pol, self.basis_modesDC),
            (qop.bI, qop.bI, qop.bI),
            np.eye(self.Ninternal),
        )
        self.qBC_ExpandAC = qop(
            ("qp", "pol", "modes"),
            (self.basis_qp, self.basis_pol, self.basis_modesAC),
            (qop.bI, qop.bI, qop.bI),
            np.eye(self.Ninternal),
        )

        # the action mode is based on which of these are available
        self.aceDC = None
        self.aceAC = None

        self.links = []

    def insert_expand_frequencies(self, ace, name):
        if self.aceAC is None:
            for fname in self.idx2fname:
                self.aceDC.insert(ace, cmn=(name, fname))
        else:
            for fname in self.idx2fname:
                self.aceAC.insert(ace, cmn=(name, fname))

    def expandDCfull(self, A, B, C, D, E):
        bSS_out = FrozenBunch(
            type="SS",
            basis="out",
            N=B.shape[-1],
        )
        bSS_in = FrozenBunch(
            type="SS",
            basis="in",
            N=C.shape[-2],
        )
        bSS_constr = FrozenBunch(
            type="SS",
            basis="states",
            N=A.shape[-2],
        )
        bSS_states = FrozenBunch(
            type="SS",
            basis="states",
            N=A.shape[-1],
        )
        qA = qop("SS", bSS_constr, bSS_states, A)
        qE = qop("SS", bSS_constr, bSS_states, E)
        qB = qop("SS", bSS_constr, bSS_in, B)
        qC = qop("SS", bSS_out, bSS_states, C)
        qD = qop("SS", bSS_out, bSS_in, D)

        qA = self.qAE_ExpandAC @ qA
        qE = self.qAE_ExpandAC @ qE
        qB = self.qAE_ExpandAC @ qB
        qC = self.qAE_ExpandAC.T @ qC
        qA.segments_set(("SS", "pol", "modes", "qp"))
        qE.segments_set(("SS", "pol", "modes", "qp"))
        qB.segments_set(("SS", "pol", "modes", "qp"))
        qC.segments_set(("SS", "pol", "modes", "qp"))
        qD.segments_set(("SS", "pol", "modes", "qp"))
        return qA.mat, qB.mat, qC.mat, qD.mat, qE.mat

    def expandAC(self, A, B, C, D, E):
        bSS_out = FrozenBunch(
            type="SS",
            basis="out",
            N=B.shape[-1],
        )
        bSS_in = FrozenBunch(
            type="SS",
            basis="in",
            N=C.shape[-2],
        )
        bSS_constr = FrozenBunch(
            type="SS",
            basis="states",
            N=A.shape[-2],
        )
        bSS_states = FrozenBunch(
            type="SS",
            basis="states",
            N=A.shape[-1],
        )
        qA = qop("SS", bSS_constr, bSS_states, A)
        qE = qop("SS", bSS_constr, bSS_states, E)
        qB = qop("SS", bSS_constr, bSS_in, B)
        qC = qop("SS", bSS_out, bSS_states, C)
        qD = qop("SS", bSS_out, bSS_in, D)

        qA = self.qAE_ExpandAC @ qA
        qE = self.qAE_ExpandAC @ qE
        qB = self.qAE_ExpandAC @ qB
        qC = self.qAE_ExpandAC.T @ qC
        qA.segments_set(("SS", "pol", "modes", "qp"))
        qE.segments_set(("SS", "pol", "modes", "qp"))
        qB.segments_set(("SS", "pol", "modes", "qp"))
        qC.segments_set(("SS", "pol", "modes", "qp"))
        qD.segments_set(("SS", "pol", "modes", "qp"))
        return qA.mat, qB.mat, qC.mat, qD.mat, qE.mat

    def addLink(self):
        return

    def connect(self, port1, port2):
        return

    def RFmodulator(self, name, frequency, mod_index):
        raise NotImplementedError()

    def space(self, name, length_m, order=5):
        """
        generate an output-only delay for a space
        """
        delta_t = length_m / c_m_s
        if self.aceDC is None:
            raise NotImplementedError()

        if self.aceAC is None:
            ABCDE = self.expandDCfull(
                A=np.array([[]]),
                B=np.array([[]]),
                C=np.array([[]]),
                D=np.eye(1),
                E=np.array([[]]),
            )
        else:
            # TODO, make the order base of the delta_t and self.linear_phase_band_Hz
            delay = dense.delay("space", delta_t, order=4)
            ABCDE = self.expandACfull(
                A=delay.A,
                B=delay.B,
                C=delay.C,
                D=delay.D,
                E=delay.E,
            )

        ace = ACE.ACE.from_ABCD(*ABCDE)
        ace2 = ACE.ACE()
        ace2.insert(ace, cmn="fw")
        ace2.insert(ace, cmn="bk")
        ace2.io_add("A-I", {("fw", "I"): None})
        ace2.io_add("B-o", {("fw", "O"): None})
        ace2.io_add("B-I", {("bk", "I"): None})
        ace2.io_add("A-o", {("bk", "O"): None})
        ace2.port_add("A", type="optical", I=("A-I"), O=("A-O"))
        ace2.port_add("B", type="optical", I=("B-I"), O=("B-O"))

        # TODO, need to apply the frequency tuning rotation
        self.insert_expand_frequencies(ace, name)
        return

    def mirror(
        self,
        name,
        R=None,
        T=None,
        k=None,
    ):
        """ """
        if self.aceDC is None:
            raise NotImplementedError()

        if T is None:
            T = 1 - R
        elif R is None:
            R = 1 - T
        r = R ** 0.5
        t = T ** 0.5

        D = np.array(
            [
                [r, t],
                [t, -r],
            ]
        )
        ABCDE = self.expandDCfull(
            A=np.array([[]]),
            B=np.array([[]]),
            C=np.array([[]]),
            D=D,
            E=np.array([[]]),
        )
        ace = ACE.ACE.from_ABCD(*ABCDE)
        ace.io_add("A-I", {("fw", "I"): None})
        ace.io_add("B-o", {("fw", "O"): None})
        ace.io_add("B-I", {("bk", "I"): None})
        ace.io_add("A-o", {("bk", "O"): None})
        ace.port_add("A", type="optical", I=("A-I"), O=("A-O"))
        ace.port_add("B", type="optical", I=("B-I"), O=("B-O"))

        if self.aceAC is None:
            return
            if P_incB is None:
                P_incB = 0
            if P_incA is None:
                P_incA = 0

            eA = P_incA ** 0.5
            eB = P_incA ** 0.5
            X = 2 * r / c_m_s

            D = (
                np.array(
                    [
                        [+r, +t, 0, 0, 0],
                        [+t, -r, 0, 0, 0],
                        [0, 0, +r, +t, R * 2 * k * eA],
                        [0, 0, +t, -r, -R * 2 * k * eB],
                        [X * eA, -X * eB, 0, 0, 0],
                    ]
                ),
            )
        ssd = ssd.in2out()
        return ssd

    def tuning(self, name, theta=None):
        """ """
        c = np.cos(theta)
        s = np.sin(theta)
        D = np.array(
            [
                [0, 0, c, -s],
                [0, 0, s, c],
                [c, -s, 0, 0],
                [s, c, 0, 0],
            ]
        )
        inputs = (
            [
                "{}+A-iQ".format(name),
                "{}+A-iP".format(name),
                "{}+B-iQ".format(name),
                "{}+B-iP".format(name),
            ],
        )
        outputs = [
            "{}+A-oQ".format(name),
            "{}+A-oP".format(name),
            "{}+B-oQ".format(name),
            "{}+B-oP".format(name),
        ]
        ssd = ssd.in2out()
        return ssd

    def freemass(
        name,
        mass_kg,
    ):
        ssd = StateSpaceDense(
            A=np.array([[0, 1], [0, 0]]),
            B=np.array([[0], [1 / mass_kg]]),
            C=np.array([[1, 0]]),
            D=np.array([[0]]),
            name=name,
            inputs=[
                "{}+M-f".format(name),
            ],
            outputs=[
                "{}+M-d".format(name),
            ],
        )
        ssd = ssd.in2out()
        return ssd
