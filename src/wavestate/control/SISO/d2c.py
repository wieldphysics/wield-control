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

from ..statespace.dense import zpk_algorithms

from . import rootset
from .rootset import SDomainRootSet
from . import siso
from . import ss

import copy
import scipy.signal


def d2c_zpk(zpk_z, fs, method='tustin', f_match=0.):
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
    Ts = 1./fs

    #set the maximum frequency for poles or zeros
    fmax = fs * 3/8.
    
    nzz, npz=len(zz), len(pz)
    zs, ps=np.zeros(nzz, dtype=np.complex), np.zeros(npz, dtype=np.complex)
    ks=kz
    
    for zidx in range(len(zz)):
        if (np.real(zz[zidx]) == -1) and (np.imag(zz[zidx]) == 0):
            zz[zidx] += 1e-16
    for pidx in range(len(pz)):
        if (np.real(pz[pidx]) == -1) and (np.imag(pz[pidx]) == 0):
            pz[pidx] += 1e-16
    
    if method.lower() == 'tustin':
        zs=(2./Ts)*(zz-1.)/(zz+1.)
        ps=(2./Ts)*(pz-1.)/(pz+1.)

        zselect = abs(zs) < fmax
        zs = zs[zselect]
        zz = zz[zselect]
        pselect = abs(ps) < fmax
        ps = ps[pselect]
        pz = pz[pselect]
        # print("select", zselect, pselect)
        nzz, npz = len(zz), len(pz)
        
        for i in range(nzz):
            ks*=(1.+zz[i])
        for i in range(npz):
            ks/=(1.+pz[i])
            
        if npz>nzz:
            zs_pad=np.ones(npz-nzz, dtype=np.complex)
            zs_pad*=(2./Ts)
            zs=np.hstack([zs, zs_pad])
            ks*=(-1)**(npz-nzz)
        elif nzz>npz:
            ps_pad=np.ones(nzz-npz, dtype=np.complex)
            ps_pad*=(2./Ts)
            ps=np.hstack([ps, ps_pad])
            ks*=(-1)**(nzz-npz)
            
    else: # use direct matching, i.e., the `matched' method in matlab
        zs=np.log(zz)/Ts
        ps=np.log(pz)/Ts
        
        __, k0=scipy.signal.freqresp((zs, ps, 1), 2.*np.pi*f_match)
        __, k1=scipy.signal.freqz_zpk(zz, pz, kz, np.pi*f_match/(fs/2.))
        ks=k1/k0
        
    return (zs, ps, ks)


