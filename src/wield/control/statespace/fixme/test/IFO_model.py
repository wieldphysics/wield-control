#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""

import numpy as np
import copy
from wield import declarative

from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.control.statespace import dense, StateSpaceDense


c_m_s = 299792458


def in2out(name, **kwargs):
    ssd = StateSpaceDense(
        A=np.array([[]]).reshape(0, 0),
        B=np.array([[]]).reshape(0, 1),
        C=np.array([[]]).reshape(1, 0),
        D=np.array([[1]]),
        name=name,
        **kwargs
    )
    return ssd


def constrain_optical(sys, nameA, nameB):
    return sys.constraints(
        "{}:{}".format(nameA, nameB),
        [
            ("bind", [nameA + "-oP", nameB + "-iP"]),
            ("bind", [nameA + "-oQ", nameB + "-iQ"]),
            ("bind", [nameA + "-iP", nameB + "-oP"]),
            ("bind", [nameA + "-iQ", nameB + "-oQ"]),
        ],
    )


def space(name, length_m, order=5):
    """
    generate an output-only delay for a space
    """
    delta_t = length_m / c_m_s
    SSAP = dense.delay("space", delta_t, order=order)
    SSAQ = copy.deepcopy(SSAP)
    SSBP = copy.deepcopy(SSAP)
    SSBQ = copy.deepcopy(SSAP)

    SSAP.names_change("inputs", fr="space.i0", to="{}+A-iP".format(name))
    SSAP.names_change("output", fr="space.o0", to="{}+B-oP".format(name))
    SSAQ.names_change("inputs", fr="space.i0", to="{}+A-iQ".format(name))
    SSAQ.names_change("output", fr="space.o0", to="{}+B-oQ".format(name))

    SSBP.names_change("inputs", fr="space.i0", to="{}+B-iP".format(name))
    SSBP.names_change("output", fr="space.o0", to="{}+A-oP".format(name))
    SSBQ.names_change("inputs", fr="space.i0", to="{}+B-iQ".format(name))
    SSBQ.names_change("output", fr="space.o0", to="{}+A-oQ".format(name))

    SSAP.names_collect("states", to="{}/ABP".format(name))
    SSAP.names_collect("constr", to="{}/ABP".format(name))
    SSAQ.names_collect("states", to="{}/ABQ".format(name))
    SSAQ.names_collect("constr", to="{}/ABQ".format(name))

    SSBP.names_collect("states", to="{}/BAP".format(name))
    SSBP.names_collect("constr", to="{}/BAP".format(name))
    SSBQ.names_collect("states", to="{}/BAQ".format(name))
    SSBQ.names_collect("constr", to="{}/BAQ".format(name))

    SS = StateSpaceDense.join(name, [SSAP, SSAQ, SSBP, SSBQ])
    SS = SS.in2out()
    return SS


def mirror(
    name,
    R=None,
    T=None,
    P_incA=None,
    P_incB=None,
    k=None,
):
    """ """
    if T is None:
        T = 1 - R
    elif R is None:
        R = 1 - T
    r = R ** 0.5
    t = T ** 0.5
    if P_incA is None and P_incB is None:
        ssd = StateSpaceDense(
            A=np.array([[]]).reshape(0, 0),
            B=np.array([[]]).reshape(0, 4),
            C=np.array([[]]).reshape(4, 0),
            D=np.array(
                [
                    [r, t, 0, 0],
                    [t, -r, 0, 0],
                    [0, 0, r, t],
                    [0, 0, t, -r],
                ]
            ),
            name=name,
            inputs=[
                "{}+A-iQ".format(name),
                "{}+B-iQ".format(name),
                "{}+A-iP".format(name),
                "{}+B-iP".format(name),
            ],
            outputs=[
                "{}+A-oQ".format(name),
                "{}+B-oQ".format(name),
                "{}+A-oP".format(name),
                "{}+B-oP".format(name),
            ],
        )
    else:
        if P_incB is None:
            P_incB = 0
        if P_incA is None:
            P_incA = 0

        eA = P_incA ** 0.5
        eB = P_incA ** 0.5
        X = 2 * r / c_m_s

        ssd = StateSpaceDense(
            A=np.array([[]]).reshape(0, 0),
            B=np.array([[]]).reshape(0, 5),
            C=np.array([[]]).reshape(5, 0),
            D=np.array(
                [
                    [+r, +t, 0, 0, 0],
                    [+t, -r, 0, 0, 0],
                    [0, 0, +r, +t, R * 2 * k * eA],
                    [0, 0, +t, -r, -R * 2 * k * eB],
                    [X * eA, -X * eB, 0, 0, 0],
                ]
            ),
            name=name,
            inputs=[
                "{}+A-iQ".format(name),
                "{}+B-iQ".format(name),
                "{}+A-iP".format(name),
                "{}+B-iP".format(name),
                "{}+M-d".format(name),
            ],
            outputs=[
                "{}+A-oQ".format(name),
                "{}+B-oQ".format(name),
                "{}+A-oP".format(name),
                "{}+B-oP".format(name),
                "{}+M-f".format(name),
            ],
        )
    ssd = ssd.in2out()
    return ssd


def tuning(name, theta=None):
    """ """
    c = np.cos(theta)
    s = np.sin(theta)
    ssd = StateSpaceDense(
        A=np.array([[]]).reshape(0, 0),
        B=np.array([[]]).reshape(0, 4),
        C=np.array([[]]).reshape(4, 0),
        D=np.array(
            [
                [0, c, 0, -s],
                [c, 0, -s, 0],
                [0, s, 0, c],
                [s, 0, c, 0],
            ]
        ),
        name=name,
        inputs=[
            "{}+A-iQ".format(name),
            "{}+B-iQ".format(name),
            "{}+A-iP".format(name),
            "{}+B-iP".format(name),
        ],
        outputs=[
            "{}+A-oQ".format(name),
            "{}+B-oQ".format(name),
            "{}+A-oP".format(name),
            "{}+B-oP".format(name),
        ],
    )
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


def print_ssd(ssd):
    print("B", ssd.B)
    print("A", ssd.A)
    print("E", ssd.E)
    print("C", ssd.C)
    print("D", ssd.D)


def build_model(
    theta=0,
    space_order=5,
    no_QRPN=False,
):
    length_m = 3995
    FSR_Hz = c_m_s / (length_m * 2)
    Sarm = space("Sarm", length_m=3995, order=space_order)
    Ssrc = space("Ssrc", length_m=56, order=space_order)
    SsrcD = tuning("SsrcD", theta=theta)

    k = 2 * np.pi / 1064e-9
    Msrm = mirror("Msrm", T=0.324, k=k)
    Mitm = mirror("Mitm", T=0.0148, k=k, P_incA=200e3)
    Metm = mirror("Metm", R=1, k=k, P_incA=200e3)
    # include the factor of 2 for the Michelson
    FMitm = freemass("FMitm", mass_kg=40 / 2)
    FMetm = freemass("FMetm", mass_kg=40 / 2)

    Ddarm = in2out("DARM")
    DsrmQ = in2out("SRMQ")
    DsrmP = in2out("SRMP")

    sys1 = StateSpaceDense.join(
        "sys",
        [
            Sarm,
            Ssrc,
            Msrm,
            Mitm,
            Metm,
            SsrcD,
            Ddarm,
            DsrmQ,
            DsrmP,
            FMitm,
            FMetm,
        ],
    )

    opc = []
    opc += constrain_optical(sys1, "Msrm+B", "SsrcD+A")
    opc += constrain_optical(sys1, "SsrcD+B", "Ssrc+A")
    opc += constrain_optical(sys1, "Ssrc+B", "Mitm+B")
    opc += constrain_optical(sys1, "Mitm+A", "Sarm+A")
    opc += constrain_optical(sys1, "Sarm+B", "Metm+A")
    sys1.outputs_delete(opc)

    sys1.constraints(
        "constraints",
        [
            (
                "zero",
                ["Metm+B-iQ"],
                ["Metm+B-iP"],
            ),
            ("bind", ["Mitm+M-d", "FMitm+M-d"]),
            ("bind", ["Msrm+A-iQ", "SRMQ.o0"]),
            ("bind", ["Msrm+A-iP", "SRMP.o0"]),
            ("sum_into", "Metm+M-d", ["DARM.o0", "FMetm+M-d"]),
        ],
    )
    if not no_QRPN:
        sys1.constraints(
            "constraintsRPN",
            [
                ("bind", ["Mitm+M-f", "FMitm+M-f"]),
                ("bind", ["Metm+M-f", "FMetm+M-f"]),
            ],
        )
    else:
        sys1.constraints(
            "constraintsRPN",
            [
                ("zero", ["FMitm+M-f"]),
                ("zero", ["FMetm+M-f"]),
            ],
        )
    return wield.bunch.Bunch(locals())


states_delays_o = [
    "Sarm/ABP",
    "Sarm/ABQ",
    "Sarm/BAP",
    "Sarm/BAQ",
    "Ssrc/ABP",
    "Ssrc/ABQ",
    "Ssrc/BAP",
    "Ssrc/BAQ",
]
states_delays_i = [
    "Sarm+A-iP",
    "Sarm+A-iQ",
    "Sarm+B-iP",
    "Sarm+B-iQ",
    "Ssrc+A-iP",
    "Ssrc+A-iQ",
    "Ssrc+B-iP",
    "Ssrc+B-iQ",
]
states_optical_i = [
    "Msrm+A-iQ",
    "Msrm+B-iQ",
    "Msrm+A-iP",
    "Msrm+B-iP",
    "Mitm+A-iQ",
    "Mitm+B-iQ",
    "Mitm+A-iP",
    "Mitm+B-iP",
    "Metm+A-iQ",
    "Metm+B-iQ",
    "Metm+A-iP",
    "Metm+B-iP",
    "SsrcD+A-iQ",
    "SsrcD+B-iQ",
    "SsrcD+A-iP",
    "SsrcD+B-iP",
]
states_mechanical = [
    "Msrm",
    "Mitm",
    "Mitm+M-d",
    "Metm",
    "Metm+M-d",
    "SsrcD",
    "DARM",
    "SRMQ",
    "SRMP",
    "FMitm",
    "FMitm+M-f",
    "FMetm",
    "FMetm+M-f",
]
