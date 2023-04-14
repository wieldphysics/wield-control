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
from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot,
    fpath_join,
    test_trigger,
    tpath_preclear,
)


from wield.utilities.np import logspaced
from wield.utilities.mpl import mplfigB

from wield.control import SISO
from wield.control.SISO import zpk_d2c_c2d


@pytest.mark.parametrize('zpk', [
    ((-1+450j, -1-450j), (-100, -100, -10), 0.01),
    ((-200+450j, -200-450j), (-100, -100, -10), 0.01),
    ((-100, -10), (-1+450j, -1-450j), 0.01),
])
def test_ZPK_c2d_various(zpk, tpath_join):
    """
    Test the conversions to and from ZPK representation and statespace representation
    using a delay filter
    """
    axB = mplfigB(Nrows=2)
    fs = 2048
    F_Hz = logspaced(1, fs/2, 1000)

    sfilt = SISO.zpk(zpk, angular=False, fiducial_rtol=1e-7)
    zfilt = zpk_d2c_c2d.c2d_zpk(sfilt, fs=fs, method='tustin')
    zfilt2 = zpk_d2c_c2d.c2d_zpk(sfilt, fs=fs, method='matched', pad=True)
    print(sfilt)
    sfiltss = sfilt.asSS
    A, B, C, D, dt = cont2discrete((sfiltss.A, sfiltss.B, sfiltss.C, sfiltss.D), 1/fs, method='tustin')
    zfiltss = SISO.statespace(A, B, C, D, dt=dt)

    print('------')
    print(zfilt)
    xfer_s1 = sfilt.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer_s1.fplot_mag, label="Direct ZPK")
    axB.ax1.semilogx(*xfer_s1.fplot_deg135)

    xfer_z1 = zfilt.fresponse(f=F_Hz)
    xfer_z2 = zfilt2.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer_z1.fplot_mag, label="Direct ZPK")
    axB.ax1.semilogx(*xfer_z1.fplot_deg135)
    axB.ax0.loglog(*xfer_z2.fplot_mag, label="Direct ZPK", ls='--')
    axB.ax1.semilogx(*xfer_z2.fplot_deg135, ls='--')

    xfer_z2 = zfiltss.fresponse(f=F_Hz)
    axB.ax0.loglog(*xfer_z2.fplot_mag, label="Statespace ZPK", ls='--')
    axB.ax1.semilogx(*xfer_z2.fplot_deg135, ls='--')

    axB.ax0.set_ylim(1e-8, 1)
    axB.save(tpath_join("test_ZPK"))

    #np.testing.assert_almost_equal(xfer4, 1/xfer4c)

def cont2discrete(system, dt, method="zoh", alpha=None):
    """
    Transform a continuous to a discrete state-space system.
    Parameters
    ----------
    system : a tuple describing the system or an instance of `lti`
        The following gives the number of elements in the tuple and
        the interpretation:
            * 1: (instance of `lti`)
            * 2: (num, den)
            * 3: (zeros, poles, gain)
            * 4: (A, B, C, D)
    dt : float
        The discretization time step.
    method : str, optional
        Which method to use:
            * gbt: generalized bilinear transformation
            * bilinear: Tustin's approximation ("gbt" with alpha=0.5)
            * euler: Euler (or forward differencing) method ("gbt" with alpha=0)
            * backward_diff: Backwards differencing ("gbt" with alpha=1.0)
            * zoh: zero-order hold (default)
            * foh: first-order hold (*versionadded: 1.3.0*)
            * impulse: equivalent impulse response (*versionadded: 1.3.0*)
    alpha : float within [0, 1], optional
        The generalized bilinear transformation weighting parameter, which
        should only be specified with method="gbt", and is ignored otherwise
    Returns
    -------
    sysd : tuple containing the discrete system
        Based on the input type, the output will be of the form
        * (num, den, dt)   for transfer function input
        * (zeros, poles, gain, dt)   for zeros-poles-gain input
        * (A, B, C, D, dt) for state-space system input
    Notes
    -----
    By default, the routine uses a Zero-Order Hold (zoh) method to perform
    the transformation. Alternatively, a generalized bilinear transformation
    may be used, which includes the common Tustin's bilinear approximation,
    an Euler's method technique, or a backwards differencing technique.
    The Zero-Order Hold (zoh) method is based on [1]_, the generalized bilinear
    approximation is based on [2]_ and [3]_, the First-Order Hold (foh) method
    is based on [4]_.
    Examples
    --------
    We can transform a continuous state-space system to a discrete one:
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cont2discrete, lti, dlti, dstep
    Define a continuous state-space system.
    >>> A = np.array([[0, 1],[-10., -3]])
    >>> B = np.array([[0],[10.]])
    >>> C = np.array([[1., 0]])
    >>> D = np.array([[0.]])
    >>> l_system = lti(A, B, C, D)
    >>> t, x = l_system.step(T=np.linspace(0, 5, 100))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(t, x, label='Continuous', linewidth=3)
    Transform it to a discrete state-space system using several methods.
    >>> dt = 0.1
    >>> for method in ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']:
    ...    d_system = cont2discrete((A, B, C, D), dt, method=method)
    ...    s, x_d = dstep(d_system)
    ...    ax.step(s, np.squeeze(x_d), label=method, where='post')
    >>> ax.axis([t[0], t[-1], x[0], 1.4])
    >>> ax.legend(loc='best')
    >>> fig.tight_layout()
    >>> plt.show()
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
    .. [2] http://techteach.no/publications/discretetime_signals_systems/discrete.pdf
    .. [3] G. Zhang, X. Chen, and T. Chen, Digital redesign via the generalized
        bilinear transformation, Int. J. Control, vol. 82, no. 4, pp. 741-754,
        2009.
        (https://www.mypolyuweb.hk/~magzhang/Research/ZCC09_IJC.pdf)
    .. [4] G. F. Franklin, J. D. Powell, and M. L. Workman, Digital control
        of dynamic systems, 3rd ed. Menlo Park, Calif: Addison-Wesley,
        pp. 204-206, 1998.
    """
    a, b, c, d = system

    if method == 'gbt':
        if alpha is None:
            raise ValueError("Alpha parameter must be specified for the "
                             "generalized bilinear transform (gbt) method")
        elif alpha < 0 or alpha > 1:
            raise ValueError("Alpha parameter must be within the interval "
                             "[0,1] for the gbt method")

    if method == 'gbt':
        # This parameter is used repeatedly - compute once here
        ima = np.eye(a.shape[0]) - alpha*dt*a
        ad = scipy.linalg.solve(ima, np.eye(a.shape[0]) + (1.0-alpha)*dt*a)
        bd = scipy.linalg.solve(ima, dt*b)

        # Similarly solve for the output equation matrices
        cd = scipy.linalg.solve(ima.transpose(), c.transpose())
        cd = cd.transpose()
        dd = d + alpha*np.dot(c, bd)

    elif method == 'bilinear' or method == 'tustin':
        return cont2discrete(system, dt, method="gbt", alpha=0.5)

    elif method == 'euler' or method == 'forward_diff':
        return cont2discrete(system, dt, method="gbt", alpha=0.0)

    elif method == 'backward_diff':
        return cont2discrete(system, dt, method="gbt", alpha=1.0)

    elif method == 'zoh':
        # Build an exponential matrix
        em_upper = np.hstack((a, b))

        # Need to stack zeros under the a and b matrices
        em_lower = np.hstack((np.zeros((b.shape[1], a.shape[0])),
                              np.zeros((b.shape[1], b.shape[1]))))

        em = np.vstack((em_upper, em_lower))
        ms = scipy.linalg.expm(dt * em)

        # Dispose of the lower rows
        ms = ms[:a.shape[0], :]

        ad = ms[:, 0:a.shape[1]]
        bd = ms[:, a.shape[1]:]

        cd = c
        dd = d

    elif method == 'foh':
        # Size parameters for convenience
        n = a.shape[0]
        m = b.shape[1]

        # Build an exponential matrix similar to 'zoh' method
        em_upper = scipy.linalg.block_diag(np.block([a, b]) * dt, np.eye(m))
        em_lower = np.zeros((m, n + 2 * m))
        em = np.block([[em_upper], [em_lower]])

        ms = scipy.linalg.expm(em)

        # Get the three blocks from upper rows
        ms11 = ms[:n, 0:n]
        ms12 = ms[:n, n:n + m]
        ms13 = ms[:n, n + m:]

        ad = ms11
        bd = ms12 - ms13 + ms11 @ ms13
        cd = c
        dd = d + c @ ms13

    elif method == 'impulse':
        if not np.allclose(d, 0):
            raise ValueError("Impulse method is only applicable"
                             "to strictly proper systems")

        ad = scipy.linalg.expm(a * dt)
        bd = ad @ b * dt
        cd = c
        dd = c @ b * dt

    else:
        raise ValueError("Unknown transformation method '%s'" % method)

    return ad, bd, cd, dd, dt
