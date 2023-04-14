"""
Examples demonstrating the use of wield.control objects to calculate
Sidles-Sigg instabilities in a number of ways.

First, starting with the state space model of the free torsional spring and
treating the hard and soft mode separately
1) Modifying the free state space model
2) Closing the radiation pressure feedback loop around the free plant
3) Using AAA to recover a zpk or state space representation from the frequency
response of an absurdly undersampled result of 1)

Second, treating the ITM and ETM together as a MIMO system at the same time and
4) Modifying the MIMO statespace with the undiagonalized stiffness matrix
5) Applying the radiation pressure feedback through a MIMO state space feedback
D matrix

Control feedback can be applied in the same way as the radiation pressure
feedback in this example is.

Provided by Kevin Kuns
"""
import numpy as np
from wield.control import SISO, MIMO
from wield.AAA import tfAAA
from wield.pytest import tpath_join, dprint, capture
from wield.bunch import Bunch
import scipy.constants as scc
from wield.control.plotting import plotTF


def test_hard_soft_modes(tpath_join, dprint, capture):
    F_Hz = np.logspace(-2, 1, 1000)

    # Parameters
    F0_Hz = 0.5
    w0_rad_s = 2 * np.pi * F0_Hz
    Q = 100
    I_kgm2 = 2.73
    Parm_W = 2e6
    Larm_m = 4e3
    Ri_m = 1934
    Re_m = 2245
    gi = 1 - Larm_m / Ri_m
    ge = 1 - Larm_m / Re_m

    # Hard and soft torsional stiffness
    k0 = 2 * Parm_W * Larm_m / (scc.c * (gi * ge - 1))
    kh = k0 * (ge + gi - np.sqrt((ge - gi)**2 + 4)) / 2
    ks = k0 * (ge + gi + np.sqrt((ge - gi)**2 + 4)) / 2

    ###########################################################################
    # SISO state space
    ###########################################################################

    # Free suspension state space matricies
    Afree_siso = np.array([
        [0, 1],
        [-w0_rad_s**2, -w0_rad_s / Q],
    ])
    B_siso = np.array([
        [0],
        [1 / I_kgm2],
    ])
    C_siso = np.array([[1, 0]])
    D_siso = np.array([[0]])

    # Hard and soft modes shift the free torsional stiffness
    Khard = np.array([
        [0, 0],
        [kh / I_kgm2, 0],
    ])
    Ksoft = np.array([
        [0, 0],
        [ks / I_kgm2, 0],
    ])
    Ahard = Afree_siso - Khard
    Asoft = Afree_siso - Ksoft

    # Make SISOStateSpace objects using the modified state space matricies
    free_siso = SISO.statespace(Afree_siso, B_siso, C_siso, D_siso)
    hard_siso = SISO.statespace(Ahard, B_siso, C_siso, D_siso)
    soft_siso = SISO.statespace(Asoft, B_siso, C_siso, D_siso)

    # Find the hard and soft mode poles
    # To find the poles, SISOStateSpace objects can be converted into ZPK
    # objects using siso.asZPK. Likewise, ZPK objects can be converted into
    # SISOStateSpace objects using zpk.asSS
    dprint('Hard poles [Hz]', hard_siso.asZPK.p / (2 * np.pi))
    dprint('Soft poles [Hz]', soft_siso.asZPK.p / (2 * np.pi))
    dprint('Soft stable?', np.all(soft_siso.asZPK.p.real < 0))

    # SISO zpk objects can also be defined directly
    # frequencies are given in rad/s by default but can be given in Hz with
    # the angular=False keyword
    Fp_Hz = np.array([
        -F0_Hz / (2 * Q) * (1 + 1j * np.sqrt(4 * Q**2 - 1)),
        -F0_Hz / (2 * Q) * (1 - 1j * np.sqrt(4 * Q**2 - 1))
    ])
    free_zpk = SISO.zpk([], Fp_Hz, 1 / I_kgm2, angular=False)
    np.testing.assert_almost_equal(
        np.sort(free_zpk.p), np.sort(free_siso.asZPK.p))

    # The hard and soft mode plants can also be found by closing the radiation
    # pressure loop: the OLG is -k * (free plant)
    # SISO objects can be added, multiplied, and divided like variables
    hard_rp_loop = (1 / (1 + kh * free_siso)) * free_siso
    soft_rp_loop = (1 / (1 + ks * free_siso)) * free_siso

    # Plot comparisons
    # Frequency response is calculated with a fresponse object. The frequency
    # vector can be specified either in Hz (with the f keyword), in rad/s (with
    # the w keyword), or in the s-domain (with the s keyword).
    # The complex numerical array is given by the tf attribute
    fig = plotTF(
        F_Hz, free_siso.fresponse(f=F_Hz).tf,
        label='Free', c='xkcd:kelly green',
    )
    plotTF(
        F_Hz, hard_siso.fresponse(f=F_Hz).tf, *fig.axes,
        label='Hard (SISO statespace)', c='xkcd:tangerine',
    )
    plotTF(
        F_Hz, soft_siso.fresponse(f=F_Hz).tf, *fig.axes,
        label='Soft (SISO statespace)', c='xkcd:burgundy',
    )
    plotTF(
        F_Hz, hard_rp_loop.fresponse(f=F_Hz).tf, *fig.axes,
        label='Hard (close RP loop)',
        ls='--', c='xkcd:royal purple',
    )
    plotTF(
        F_Hz, soft_rp_loop.fresponse(f=F_Hz).tf, *fig.axes,
        label='Soft (close RP loop)',
        ls='--', c='xkcd:sky blue',
    )
    fig.axes[1].legend(loc='upper left')
    fig.set_size_inches((6, 6.4))
    fig.savefig(tpath_join('compare_siso.pdf'))

    ###########################################################################
    # AAA fit
    ###########################################################################

    # Generate some absurdly undersampled data to fit
    F_fit_Hz = np.logspace(-1, 1, 5)
    hard_fit_data = hard_siso.fresponse(f=F_fit_Hz).tf
    soft_fit_data = soft_siso.fresponse(f=F_fit_Hz).tf

    # Use AAA to fit this data
    hard_fit = tfAAA(F_fit_Hz, hard_fit_data)
    soft_fit = tfAAA(F_fit_Hz, soft_fit_data)

    # Define a zpk object from these fits
    # AAA uses the IIRrational normalization, which needs to be specified
    # The standard normalization ('scipy') is the default
    hard_aaa = SISO.zpk(*hard_fit.zpk, convention='iirrational')
    soft_aaa = SISO.zpk(*soft_fit.zpk, convention='iirrational')

    # Check that the fit recovered the poles of the orginal SISO model
    np.testing.assert_almost_equal(
        np.sort(hard_siso.asZPK.p), np.sort(hard_aaa.p))
    np.testing.assert_almost_equal(
        np.sort(soft_siso.asZPK.p), np.sort(soft_aaa.p))

    # Plot the data on top of the fits
    fig = plotTF(
        F_Hz, hard_aaa.fresponse(f=F_Hz).tf,
        label='Hard AAA fit', c='xkcd:cerulean',
    )
    plotTF(
        F_Hz, soft_aaa.fresponse(f=F_Hz).tf, *fig.axes,
        label='Soft AAA fit', c='xkcd:tangerine',
    )
    plotTF(
        F_fit_Hz, hard_fit_data, *fig.axes,
        label='Hard AAA fit data',
        ls='', marker='o', c='xkcd:cerulean')
    plotTF(
        F_fit_Hz, soft_fit_data, *fig.axes,
        label='Soft AAA fit data',
        ls='', marker='o', c='xkcd:tangerine',
    )
    fig.axes[1].legend(loc='upper left')
    fig.axes[0].set_xlim(F_Hz[0], F_Hz[-1])
    fig.set_size_inches((6, 6.4))
    fig.savefig(tpath_join('compare_aaa.pdf'))

    ###########################################################################
    # MIMO state space
    ###########################################################################

    # Free suspension state space matrices
    # State variables are
    # (ETM angle, ITM angle, ETM angular velocity, ITM angular velocity)
    eye = np.eye(2)
    Afree_mimo = np.block([
        [0 * eye, eye],
        [-w0_rad_s**2 * eye, -w0_rad_s / Q * eye],
    ])
    B_mimo = np.block([
        [0 * eye],
        [1 / I_kgm2 * eye],
    ])
    C_mimo = np.block([[eye, 0 * eye]])
    D_mimo = 0 * eye

    # RP torsional stiffness matrix
    Krp = k0 * np.array([
        [gi, 1],
        [1, ge],
    ])
    K_mimo = np.block([
        [0 * eye, 0 * eye],
        [Krp / I_kgm2, 0 * eye],
    ])
    A_mimo = Afree_mimo - K_mimo

    # Make MIMOStateSpace objects
    # Input/output degrees of freedom/test points are defined by either lists
    # or dictionaries specifying the row/column indices to which they
    # correspond. They can have different names but are the same in this example.
    dofs = {'etm': 0, 'itm': 1}
    # dofs = ['etm', 'itm']
    free_mimo = MIMO.statespace(
        Afree_mimo, B_mimo, C_mimo, D_mimo,
        inputs=dofs,
        outputs=dofs,
    )
    ss_mimo = MIMO.statespace(
        A_mimo, B_mimo, C_mimo, D_mimo,
        inputs=dofs,
        outputs=dofs,
    )

    # The radiation pressure modified state space can also be defined by
    # specifying feedback connections in the free model to define a feedback
    # D matrix
    connections = {
        ('etm', 'etm'): -Krp[0, 0],
        ('etm', 'itm'): -Krp[0, 1],
        ('itm', 'etm'): -Krp[1, 0],
        ('itm', 'itm'): -Krp[1, 1],
    }
    fback_ss = free_mimo.feedback_connect(connections=connections)

    # These MIMOStateSpace objects are in the mirror basis. (Choosing different
    # B and C matrices could have put them in the hard/soft basis instead, in
    # which case the dofs dictionary should have been changed to have keys
    # 'hard' and 'soft' instead).
    # Individual SISOStateSpace objects can be extracted like
    free_etm_siso = free_mimo.siso('etm', 'etm')

    # # MIMO fresponse (in the mirror basis) is calculated like SISO is.
    # # SISO fresponse can also be extracted from these MIMO fresponse objects
    mimo_response = ss_mimo.fresponse(f=F_Hz)
    fback_response = fback_ss.fresponse(f=F_Hz)

    # Plot results in the mirror basis
    fig = plotTF(
        F_Hz, free_etm_siso.fresponse(f=F_Hz).tf,
        label='Free', c='xkcd:kelly green',
    )
    plotTF(
        F_Hz, mimo_response.siso('etm', 'etm').tf, *fig.axes,
        label='ETM to ETM', c='xkcd:cerulean',
    )
    plotTF(
        F_Hz, mimo_response.siso('etm', 'itm').tf, *fig.axes,
        label='ITM to ETM', c='xkcd:tangerine',
    )
    fig.axes[1].legend(loc='upper left')
    fig.set_size_inches((6, 6.4))
    fig.savefig(tpath_join('mimo_mirror_basis.pdf'))

    # Analyze results in the hard/soft basis.
    # The full numerical MIMO plant (now a matrix) is given by the tf attribute
    # The hard and soft eigenvectors are
    _, eigv = np.linalg.eig(Krp)
    vhard = eigv[:, 0]
    vsoft = eigv[:, 1]

    # Convert to hard/soft basis
    def to_hard_soft(fresponse):
        return Bunch(
            hard = vhard.T @ fresponse.tf @ vhard,
            soft = vsoft.T @ fresponse.tf @ vsoft,
        )

    mimo_plants = to_hard_soft(mimo_response)
    fback_plants = to_hard_soft(fback_response)

    # Plot comparisons
    fig = plotTF(
        F_Hz, free_etm_siso.fresponse(f=F_Hz).tf,
        label='Free', c='xkcd:kelly green',
    )
    plotTF(
        F_Hz, mimo_plants.hard, *fig.axes,
        label='Hard (MIMO statespace)', c='xkcd:tangerine',
    )
    plotTF(
        F_Hz, mimo_plants.soft, *fig.axes,
        label='Soft (MIMO statespace)', c='xkcd:burgundy',
    )
    plotTF(
        F_Hz, fback_plants.hard, *fig.axes,
        label='Hard (MIMO feedback)',
        ls='--', c='xkcd:royal purple',
    )
    plotTF(
        F_Hz, fback_plants.soft, *fig.axes,
        label='Soft (MIMO feedback)',
        ls='--', c='xkcd:sky blue',
    )
    fig.axes[1].legend(loc='upper left')
    fig.set_size_inches((6, 6.4))
    fig.savefig(tpath_join('compare_mimo.pdf'))

    ###########################################################################
    # Check that all of these methods are equavalent
    ###########################################################################

    def assert_equal(hard, soft):
        np.testing.assert_almost_equal(
            hard_siso.fresponse(f=F_Hz).tf,
            hard,
        )
        np.testing.assert_almost_equal(
            soft_siso.fresponse(f=F_Hz).tf,
            soft,
        )

    assert_equal(
        hard_rp_loop.fresponse(f=F_Hz).tf,
        soft_rp_loop.fresponse(f=F_Hz).tf,
    )
    assert_equal(
        hard_aaa.fresponse(f=F_Hz).tf,
        soft_aaa.fresponse(f=F_Hz).tf,
    )
    assert_equal(
        mimo_plants.hard,
        mimo_plants.soft,
    )
    assert_equal(
        fback_plants.hard,
        fback_plants.soft,
    )
