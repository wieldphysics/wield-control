#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions to create a SISO state space system from inputs.
"""
import numbers
import numpy as np
import warnings

from ..algorithms.statespace.dense import zpk_algorithms

from ..utilities import algorithm_choice

from . import ss

def zpk2ss_cheby(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    if len(z) <= len(p) and len(p) > 0:
        ABCDE = zpk_algorithms.zpk_rc(
            Zc=z.c_plane,
            Zr=z.r_line,
            Pc=p.c_plane,
            Pr=p.r_line,
            k=zpk.k,
            convention="scipy",
            orientation="upper",
            method='companion_cheby',
            # method='chain_poly',
        )
    else:
        raise ValueError("Cheby ZPK cannot handle more poles than zeros")

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )

    # statesp = statesp.balance_and_truncate(nr=statesp.ss.Nstates-0)
    return statesp

algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_cheby', zpk2ss_cheby, 80)


def zpk2ss_chain_poly(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    ABCDE = zpk_algorithms.zpk_rc(
        Zc=z.c_plane,
        Zr=z.r_line,
        Pc=p.c_plane,
        Pr=p.r_line,
        k=zpk.k,
        convention="scipy",
        orientation="upper",
        method='chain_poly',
    )

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )
    # statesp = statesp.balance_and_truncate(nr=statesp.ss.Nstates-0)
    return statesp


algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_chain_poly', zpk2ss_chain_poly, 100)


def zpk2ss_chainE_poly(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    ABCDE = zpk_algorithms.zpk_rc(
        Zc=z.c_plane,
        Zr=z.r_line,
        Pc=p.c_plane,
        Pr=p.r_line,
        k=zpk.k,
        convention="scipy",
        orientation="upper",
        method='chainE_poly',
    )

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )
    # statesp = statesp.reduceE()
    # statesp = statesp.reduceE2()
    return statesp


algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_chainE_poly', zpk2ss_chainE_poly, 60)

def zpk2ss_cheby_lower(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    if len(z) <= len(p) and len(p) > 0:
        ABCDE = zpk_algorithms.zpk_rc(
            Zc=z.c_plane,
            Zr=z.r_line,
            Pc=p.c_plane,
            Pr=p.r_line,
            k=zpk.k,
            convention="scipy",
            orientation="lower",
            method='companion_cheby',
            # method='chain_poly',
        )
    else:
        raise ValueError("Cheby ZPK cannot handle more poles than zeros")

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )
    # statesp = statesp.balance_and_truncate(nr=statesp.ss.Nstates-0)
    return statesp

algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_cheby_lower', zpk2ss_cheby_lower, -80)


def zpk2ss_chain_poly_lower(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    ABCDE = zpk_algorithms.zpk_rc(
        Zc=z.c_plane,
        Zr=z.r_line,
        Pc=p.c_plane,
        Pr=p.r_line,
        k=zpk.k,
        convention="scipy",
        orientation="lower",
        method='chain_poly',
    )

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )
    # statesp = statesp.balance_and_truncate(nr=statesp.ss.Nstates-0)
    return statesp


algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_chain_poly_lower', zpk2ss_chain_poly_lower, -100)


def zpk2ss_chainE_poly_lower(zpk):
    assert (zpk.hermitian)

    z = zpk.zeros.drop_mirror_imag()
    p = zpk.poles.drop_mirror_imag()
    # Currently only supports hermitian inputs

    ABCDE = zpk_algorithms.zpk_rc(
        Zc=z.c_plane,
        Zr=z.r_line,
        Pc=p.c_plane,
        Pr=p.r_line,
        k=zpk.k,
        convention="scipy",
        orientation="lower",
        method='chainE_poly',
    )

    statesp = ss.statespace(
        ABCDE,
        hermitian=zpk.hermitian,
        time_symm=zpk.time_symm,
        fiducial=zpk.fiducial,
        fiducial_rtol=zpk.fiducial_rtol,
        fiducial_atol=zpk.fiducial_atol,
        algorithm_choices=zpk.algorithm_choices,
        algorithm_ranking=zpk.algorithm_ranking,
        # flags={"schur_real_upper", "hessenburg_upper"},
    )
    # return statesp
    # statesp = statesp.reduceE2()
    # statesp = statesp.balance_and_truncate()
    return statesp


algorithm_choice.algorithm_register('zpk2ss', 'zpk2ss_chainE_poly_lower', zpk2ss_chainE_poly_lower, -60)

