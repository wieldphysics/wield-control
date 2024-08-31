import os
import contextlib

import numpy as np
import matplotlib.pyplot as plt
import control

import pytest
import scipy
from scipy.signal import cheby2
from scipy import signal

from wield.bunch import Bunch
#from wield.control import MIMO
from wield.control import SISO
from wield.control.ss_bare.ss import BareStateSpace as RawStateSpace
from wield.control.ss_bare.ssprint import print_dense_nonzero

import altair as alt
import pandas as pd

def zp_frequencies(z,p):
    z = z[z.imag >= 0]
    p = p[p.imag >= 0]

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

    return F_include

def bode(
    sys,
    F_Hz=None,
    omega_limits=None,
    include_zp = True,
    label="SISO system",
):
    """
    sys should be a wield.control.SISO object
    """

    if include_zp:
        z, p = sys._zp
        F_include = zp_frequencies(z,p)

        if omega_limits is None:
            omega_limits = (
                (F_include[0] + 1e-6) / 3,
                F_include[-1] * 3
            )

    if F_Hz is None:
        F_Hz = np.geomspace(omega_limits[0], omega_limits[1], 1000)

    if include_zp:
        F_Hz = np.sort(np.concatenate([F_Hz, F_include]))

    fr = sys.fresponse(f=F_Hz)

    alldata = pd.DataFrame({'frequency (Hz)':fr.f,'magnitude':fr.mag,'phase':fr.deg,'label':label})

    magnitudechart = alt.Chart(alldata).mark_line().encode(
    x=alt.X('frequency (Hz):Q').scale(type="log"),
    y=alt.Y('magnitude:Q').scale(type="log"),
    color='label:N',
    ).properties(
        width=400,
        height=200
    )

    phasechart = alt.Chart(alldata).mark_line().encode(
        x=alt.X('frequency (Hz):Q').scale(type="log"),
        y=alt.Y('phase:Q'),
        color='label:N',
    ).properties(
        width=400,
        height=100
    )

    chart = alt.vconcat(magnitudechart,phasechart)

    return chart

#@contextlib.contextmanager
def multi_bode(syslist,
    labels = None,
    F_Hz = None,
    omega_limits=None,
    include_zp = True,
):
    """
    Plot multiple bode plots at once.

    The main advantage is that the scale of the axes can be better auto-determined to capture all known poles and zeros.
    """

    zlist = np.array([])
    plist = np.array([])

    if include_zp:
        for sys in syslist:
            z, p = sys._zp
            zlist = np.concatenate((zlist,z))
            plist = np.concatenate((plist,p))

        F_include = zp_frequencies(zlist,plist)

        if omega_limits is None:
            omega_limits = (
                (F_include[0] + 1e-6) / 3,
                F_include[-1] * 3
            )

    if F_Hz is None:
        F_Hz = np.geomspace(omega_limits[0], omega_limits[1], 1000)

    #this makes include_zp irrelevant if F_Hz or omega_limits are passed
    F_include = F_include[F_include < F_Hz[-1]]
    F_include = F_include[F_include > F_Hz[0]]

    if include_zp:
        F_Hz = np.sort(np.concatenate([F_Hz, F_include]))

    fresponses = [sys.fresponse(f=F_Hz) for sys in syslist]
    
    if labels == None:
        labels = [f"system {ii}" for ii in range(len(syslist))]

    dataframes = [pd.DataFrame({'frequency (Hz)':fr.f,'magnitude':fr.mag,'phase':fr.deg,'label':label}) for (fr,label) in zip(fresponses,labels)]
    
    alldata = pd.concat(dataframes)

    magnitudechart = alt.Chart(alldata).mark_line().encode(
    x=alt.X('frequency (Hz):Q').scale(type="log"),
    y=alt.Y('magnitude:Q').scale(type="log"),
    color='label:N',
    ).properties(
        width=400,
        height=200
    )

    phasechart = alt.Chart(alldata).mark_line().encode(
        x=alt.X('frequency (Hz):Q').scale(type="log"),
        y=alt.Y('phase:Q'),
        color='label:N',
    ).properties(
        width=400,
        height=100
    )

    chart = alt.vconcat(magnitudechart,phasechart)

    return chart