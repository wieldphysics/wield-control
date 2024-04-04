"""
Unit tests and examples for damping the LIGO QUAD suspensions

Provided by Kevin Kuns
"""
import numpy as np
import pytest

from wield.control import SISO, MIMO
from wield.bunch import Bunch
from wield.pytest import fjoin, tjoin, dprint
from wield.control.plotting import plotTF
from wield.utilities.file_io import load, save


def test_convert_zpk_ss():
    """
    Converts a LIGO damping filter from ZPK to statespace and back to test numerical stability
    """
    fname = fjoin('damping_filters.h5')
    zpk = load(fname).L
    dprint('testing!')
    filt_zpk = SISO.zpk(zpk.z, zpk.p, zpk.k)
    filt_ss = filt_zpk.asSS
    filt_zpk2 = filt_ss.asZPK

    # TODO - no reason to think these sets are sorted the same way for a direct comparison
    dprint(np.allclose(filt_zpk.z, filt_zpk2.z))
    # TODO - no reason to think these sets are sorted the same way for a direct comparison
    dprint(np.allclose(filt_zpk.p, filt_zpk2.p))
    # this should be True
    dprint(np.isclose(filt_zpk.k, filt_zpk2.k))


@pytest.mark.parametrize('quad_type', ['quad_full', 'quad_small'])
def test_damp_quad(quad_type):
    """
    Damp the undamped LIGO QUAD state space model

    Tests calculation of frequency responses and conversion of the resulting damped state space
    plant to ZPK
    """
    # load quad state space
    ss_data = load(fjoin(quad_type + '.h5'))
    udamp_plant = MIMO.MIMOStateSpace(
        ss_data.A, ss_data.B, ss_data.C, ss_data.D,
        inputs=dict(ss_data.inputs), outputs=dict(ss_data.outputs),
    )

    # load ZPK damping filters and convert to state space
    zpk_data = load(fjoin('damping_filters.h5'))
    damp_filts = Bunch()
    for dof, zpk in zpk_data.items():
        damp_filts[dof] = SISO.zpk(zpk.z, zpk.p, zpk.k).asSS

    # Need to make the SISO damping filters into 1-1 MIMO filters with specified
    # inputs and outputs for making the feedback connections
    damp_plant = MIMO.ssjoinsum(
        udamp_plant,
        damp_filts.L.mimo('M0.L.o', 'M0.L.i'),
        damp_filts.T.mimo('M0.T.o', 'M0.T.i'),
        damp_filts.V.mimo('M0.V.o', 'M0.V.i'),
        damp_filts.P.mimo('M0.P.o', 'M0.P.i'),
        damp_filts.Y.mimo('M0.Y.o', 'M0.Y.i'),
        damp_filts.R.mimo('M0.R.o', 'M0.R.i'),
    )
    # connect the displacements to the inputs to the damping filters
    # and then connect the outputs from the filters to the drives
    connections = [
        ('M0.L.i', 'M0.disp.L'), ('M0.drive.L', 'M0.L.o'),
        ('M0.T.i', 'M0.disp.T'), ('M0.drive.T', 'M0.T.o'),
        ('M0.V.i', 'M0.disp.V'), ('M0.drive.V', 'M0.V.o'),
        ('M0.P.i', 'M0.disp.P'), ('M0.drive.P', 'M0.P.o'),
        ('M0.Y.i', 'M0.disp.Y'), ('M0.drive.Y', 'M0.Y.o'),
        ('M0.R.i', 'M0.disp.R'), ('M0.drive.R', 'M0.R.o'),
    ]
    # make the feedback and balance for numerical stability
    damp_plant = damp_plant.feedback_connect(connections=connections)
    udamp_plant = udamp_plant.balance()
    damp_plant = damp_plant.balance()

    # plot some transfer functions
    F_Hz = np.geomspace(0.05, 10, 1000)
    dprint('Calculating undamped L3')
    udamp_fresp_L3 = udamp_plant.siso('L3.disp.L', 'L3.drive.L').fresponse(f=F_Hz).tf
    dprint('Calculating damped L3')
    damp_fresp_L3 = damp_plant.siso('L3.disp.L', 'L3.drive.L').fresponse(f=F_Hz).tf
    dprint('Calculating undamped M0')
    udamp_fresp_M0 = udamp_plant.siso('M0.disp.L', 'M0.drive.L').fresponse(f=F_Hz).tf
    dprint('Calculating damped M0')
    damp_fresp_M0 = damp_plant.siso('M0.disp.L', 'M0.drive.L').fresponse(f=F_Hz).tf

    fig = plotTF(F_Hz, udamp_fresp_L3, label='Undamped')
    plotTF(F_Hz, damp_fresp_L3, *fig.axes, label='Damped', ls='--')
    fig.axes[0].legend()
    fig.axes[0].set_title('L3 to L3')
    fig.savefig(tjoin('L3.pdf'))

    fig = plotTF(F_Hz, udamp_fresp_M0, label='Undamped')
    plotTF(F_Hz, damp_fresp_M0, *fig.axes, label='Damped', ls='--')
    fig.axes[0].legend()
    fig.axes[0].set_title('M0 to M0')
    fig.savefig(tjoin('M0.pdf'))

    # make zpk filters and convert back to SS
    for (to, fr) in zip(['L3.disp.L', 'M0.disp.L'], ['L3.drive.L', 'M0.drive.L']):
        siso_udamp = damp_plant.siso(to, fr)
        siso_damp = udamp_plant.siso(to, fr)
        zpk_udamp = siso_udamp.asZPK
        zpk_damp = siso_damp.asZPK
        zpk_udamp.asSS
        zpk_damp.asSS
