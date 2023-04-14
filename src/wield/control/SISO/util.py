#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import warnings
from . import response


class NumericalWarning(UserWarning):
    pass


def build_fiducial(
    fiducial=None,
    fiducial_w=None,
    fiducial_f=None,
    fiducial_s=None,
    fiducial_z=None,
    dt=None,
):
    if fiducial is not None:
        if isinstance(fiducial, response.SISOFResponse):
            pass
        elif callable(fiducial):
            assert(fiducial_f is None)
            assert(fiducial_w is None)
            assert(fiducial_s is None)
            assert(fiducial_z is None)
        else:
            fiducial = np.asarray(fiducial)
            fiducial = response.SISOFResponse(
                f=fiducial_f,
                w=fiducial_w,
                s=fiducial_s,
                z=fiducial_z,
                tf=fiducial,
                dt=dt,
            )
    else:
        if (
                (fiducial_f is not None)
                or (fiducial_w is not None)
                or (fiducial_s is not None)
                or (fiducial_z is not None)
        ):
            fiducial = response.SISOFResponse(
                f=fiducial_f,
                w=fiducial_w,
                s=fiducial_s,
                z=fiducial_z,
                tf=None,
                dt=dt,
            )
    return fiducial


def build_sorz(
    f=None,
    w=None,
    s=None,
    z=None,
    dt=None,
):
    sorz = None
    if dt is None:
        if f is not None:
            sorz = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(sorz is None)
            sorz = 1j * np.asarray(w)
        if s is not None:
            assert(sorz is None)
            sorz = np.asarray(s)
        assert(z is None)
    else:
        if f is not None:
            if np.max(f) > 1/dt/2:
                warnings.warn(f"frequency response evaluated above the f_Nyquist {1/dt/2}. Expect aliasing.")
            sorz = np.exp(2j * np.pi * np.asarray(f) * dt)
        if w is not None:
            if np.max(w)/(2 * np.pi) > 1/dt/2:
                warnings.warn(f"frequency response evaluated above the f_Nyquist {1/dt/2}. Expect aliasing.")
            assert(sorz is None)
            sorz = np.exp(1j * np.asarray(w) * dt)
        if z is not None:
            assert(sorz is None)
            sorz = np.asarray(z)
        assert(s is None)
    return sorz
