#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Perform discrete to continuous conversion
"""
import numbers
import numpy as np
import itertools

from . import zpk

import copy
import scipy.signal


def d2c_zpk(zpk_z, fs, method="tustin"):
    """
    # From pyctrl, .../userapps../lsc/h1/scripts/feedforward/ipython_notebooks/pyctrl.py
    ## discrete <-> continuous conversion
    cf. https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html#bs78nig-12

    Updated by Lee McCuller 2022Jun to purge spuriously large roots with the Tustin method.
    Previously some filters would create
    roots at 1e15Hz, which would then stress the numerical precision of the gain.
    """
    zpk = copy.deepcopy(zpk_z)
    zz, pz, kz = zpk
    dt = 1.0 / fs
    fs2x = 2 * fs

    # set the maximum frequency for poles or zeros
    fmax = fs * 3 / 8.0

    nzz, npz = len(zz), len(pz)
    zs, ps = np.zeros(nzz, dtype=np.complex), np.zeros(npz, dtype=np.complex)
    ks = kz

    method = method.lower()
    if method == "tustin":
        zs = fs2x * (zz - 1.0) / (zz + 1.0)
        ps = fs2x * (pz - 1.0) / (pz + 1.0)

        zselect = abs(zs) < fmax
        zs = zs[zselect]
        zz = zz[zselect]
        pselect = abs(ps) < fmax
        ps = ps[pselect]
        pz = pz[pselect]
        # print("select", zselect, pselect)
        nzz, npz = len(zz), len(pz)

        for i in range(nzz):
            # kz /= 1.0 - 0.5 * dt * zs[i]
            ks *= 1.0 + zz[i]
        for i in range(npz):
            # kz *= 1.0 - 0.5 * dt * zs[i]
            ks /= 1.0 + pz[i]

        if npz > nzz:
            zs_pad = fs2x * np.ones(npz - nzz, dtype=np.complex)
            zs = np.hstack([zs, zs_pad])
            ks *= (-1)**(npz - nzz)
        elif nzz > npz:
            ps_pad = fs2x * np.ones(nzz - npz, dtype=np.complex)
            ps = np.hstack([ps, ps_pad])
            ks *= (-1)**(nzz - npz)

    else:  # use direct matching, i.e., the `matched' method in matlab
        zs = np.log(zz) / dt
        ps = np.log(pz) / dt

        __, k0 = scipy.signal.freqresp((zs, ps, 1), 2.0 * np.pi * f_match)
        __, k1 = scipy.signal.freqz_zpk(zz, pz, kz, np.pi * f_match / (fs / 2.0))
        ks = k1 / k0

    return (zs, ps, ks)


def c2d_zpk(zpk_s, *, method, dt=None, fs=None, pad=None):
    assert(zpk_s.dt is None)

    if dt is None:
        dt = 1 / fs
    else:
        assert(fs is None)
        fs = 1 / dt

    assert(fs > 0)

    zs = zpk_s.z
    ps = zpk_s.p
    ks = zpk_s.k

    nzs = len(zs)
    nps = len(ps)
    zz = np.zeros(nzs, dtype=np.complex)
    pz = np.zeros(nps, dtype=np.complex)

    kz = ks

    # currently must fix this at one since the gain calculation doesn't compensate
    # for GBT != 1/2
    gbt_num = 0.5
    gbt_den = 1 - gbt_num
    kadj = 1

    method = method.lower()
    if method == "tustin":
        zz = (1.0 + gbt_num * dt * zs) / (1.0 - gbt_den * dt * zs)
        pz = (1.0 + gbt_num * dt * ps) / (1.0 - gbt_den * dt * ps)

    elif method == "matched":
        zz = np.exp(zs * dt)
        pz = np.exp(ps * dt)

    else:
        raise RuntimeError("Unrecognized c2d type {} for converting zpk's")

    for i, j in itertools.zip_longest(range(nzs), range(nps)):
        # the use of itertools zip interlaces the roots, so that
        # we don't accidentally exhaust the range of the kadj gain float

        # if the pole or zero is this close, then the gain isn't affected anyway
        if i is not None and (abs(zs[i]) / fs) > 1e-12:
            kadj /= (-1.0 + zz[i]) / zs[i]

        if j is not None and (abs(ps[j]) / fs) > 1e-12:
            kadj *= (-1.0 + pz[j]) / ps[j]

    if pad is None:
        if nps > nzs:
            pad = True
        else:
            pad = False
    if pad:
        kadj *= (2.0 - 1/fs)**float(nzs - nps)
        if nps > nzs:
            zz_pad = -np.ones(nps - nzs, dtype=np.complex) * (1 - 1 / fs)
            zz = np.hstack([zz, zz_pad])
        elif nzs > nps:
            pz_pad = -np.ones(nzs - nps, dtype=np.complex) * (1 - 1 / fs)
            pz = np.hstack([pz, pz_pad])

    kz *= kadj
    if zpk_s.hermitian:
        kz = kz.real
    return zpk(zz, pz, kz, dt = dt)
