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
import scipy
import scipy.signal

import pytest
import contextlib

from wield.pytest import (  # noqa: F401
    tjoin,
    fjoin,
    dprint,
)


from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB
from wield.bunch import Bunch

from wield.control import MIMO, SISO
from wield.utilities.file_io import load, save


c_m_s = 299792458


def test_load_convert():
    axB = mplfigB(Nrows=1, Ncols=1)
    with multi_bode(axB=axB) as bode:
        for fname in ['K.mat',
                      'Kp.mat',
                      'KpPpPinv.mat',
                      'P.mat',
                      'Pp.mat',]:
            m = Bunch(load(fjoin(fname)))
            ss = SISO.SISOStateSpace(A=m.A, B=m.B.reshape(-1, 1), C=m.C.reshape(1, -1), D=np.asarray(m.D).reshape(1, 1),)
            # dprint(ss)
            zpk = ss.asZPK
            # import warnings
            # warnings.filterwarnings("error")
            print(zpk.z, zpk.p, zpk.k)
            bode(ss)
            bode(zpk, ls='--')

    axB.save(tjoin('bode'))
    return


def bode(
    sys,
    axB=None,
    F_Hz=None,
    omega_limits=None,
    include_zp = True,
    label=None,
    **kwargs
):
    """
    sys should be a wield.control.SISO object
    """
    if axB is None:
        axB = mplfigB(Nrows = 2)

    if include_zp:
        z, p = sys._zp
        z = z[z.imag > 0]
        p = p[p.imag > 0]
        F_include = np.sort(np.concatenate(
            [
                z.imag,
                p.imag,
                z.imag - abs(z.real) - 1e-6,
                z.imag + abs(z.real) + 1e-6,
                p.imag - abs(p.real) - 1e-6,
                p.imag + abs(p.real) + 1e-6,
            ]
        )) / (2 * np.pi)
        F_include = F_include[F_include > 0]

        if omega_limits is None:
            omega_limits = (
                (F_include[0] + 1e-6) / 3,
                F_include[-1] * 3
            )

    if F_Hz is None:
        F_Hz = np.geomspace(omega_limits[0], omega_limits[1], 1000)

    if include_zp:
        print("F_include", F_include)
        F_Hz = np.sort(np.concatenate([F_Hz, F_include]))

    fr = sys.fresponse(f=F_Hz)
    axB.ax0.loglog(*fr.fplot_mag, label=label, **kwargs)
    if hasattr(axB, 'ax1'):
        axB.ax1.semilogx(*fr.fplot_deg225, label=label, **kwargs)
    return axB


@contextlib.contextmanager
def multi_bode(
    axB,
    F_Hz = None,
    include_zp = True,
    **kwargs
):
    """
    Plot multiple bode plots at once.

    The main advantage is that the scale of the axes can be better auto-determined to capture all known poles and zeros.
    The input arguments should all be dictionaries containing the bode kwargs.

    the usage should be

    the kwargs on the top call are passed into all sub calls

    with multi_bode(axB=axB, **kw_cmn) as bode:
        bode(sisoA, label='A', **kw)
        bode(sisoB, label='B', **kw)
        bode(sisoC, label='C', **kw)
    axB.save('figname.pdf')
    """
    bode_arg_kwarg_list = []
    F_includes = []

    def bode_sys(
        sys,
        **kwargs
    ):
        """
        Just extract and return the bode system
        """
        return sys

    def bode_plot(*args, **kwargs):
        sys = bode_sys(*args, **kwargs)

        z, p = sys._zp
        z = z[z.imag > 0]
        p = p[p.imag > 0]
        F_include = np.sort(np.concatenate(
            [
                z.imag,
                p.imag,
                z.imag - abs(z.real) - 1e-6,
                z.imag + abs(z.real) + 1e-6,
                p.imag - abs(p.real) - 1e-6,
                p.imag + abs(p.real) + 1e-6,
            ]
        )) / (2 * np.pi)
        F_include = F_include[F_include > 0]
        F_includes.append(F_include)

        bode_arg_kwarg_list.append(
            (args, kwargs)
        )

    yield bode_plot

    F_include = np.sort(np.concatenate(F_includes))

    if F_Hz is None:
        F_Hz = np.geomspace(
            (F_include[0] + 1e-6) / 3,
            F_include[-1] * 3,
            300
        )

    F_include = F_include[F_include < F_Hz[-1]]
    F_include = F_include[F_include > F_Hz[0]]

    if include_zp:
        F_Hz = np.sort(np.concatenate([F_Hz, F_include]))

    for arg, kwarg in bode_arg_kwarg_list:
        kw = dict(**kwargs)
        kw.update(kwarg)
        bode(
            *arg,
            axB=axB,
            F_Hz=F_Hz,
            omega_limits=None,
            include_zp = False,
            **kw
        )
    return

