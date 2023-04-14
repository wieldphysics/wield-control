import numpy as np
import matplotlib.pyplot as plt


def plotTF(ff, tf, mag_ax=None, phase_ax=None, **kwargs):
    """
    Plot a SISO transfer function

    Parameters
    ----------
    ff : frequency array
    tf : complex transfer function array
    mag_ax : optional existing axis to plot the magnitude on
    phase_ax : optional existing axis to plot the phase on
    kwargs : matplotlib keyword arguments

    Returns
    -------
    fig : transfer function figure if mag_ax and phase_ax are not given

    Examples
    --------
    Plot two transfer functions on the same plot
    >>> fig = plotTF(F_Hz, tf1, label='TF1')
    >>> plotTF(F_Hz, tf2, *fig.axes, label='TF2', ls='--')
    >>> fig.axes[0].legend()
    """
    if not(mag_ax and phase_ax):
        if (mag_ax is not None) or (phase_ax is not None):
            msg = 'If one of the phase or magnitude axes is given,'
            msg += ' the other must be given as well.'
            raise ValueError(msg)
        newFig = True
    else:
        newFig = False

    if newFig:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
        mag_ax = fig.add_subplot(gs[0])
        phase_ax = fig.add_subplot(gs[1], sharex=mag_ax)

    mag_ax.loglog(ff, np.abs(tf), **kwargs)
    mag_ax.set_ylabel('Magnitude')
    mag_ax.autoscale(enable=True, axis='y')

    # If the TF is close to being constant magnitude, increase ylims
    # in order to show y tick labels and avoid a misleading plot.
    ylim = mag_ax.get_ylim()
    if ylim[1] / ylim[0] < 10:
        mag_ax.set_ylim(ylim[0] / 10.1, ylim[1] * 10.1)

    mag_ax.set_xlim(min(ff), max(ff))
    phase_ax.set_ylim(-185, 185)
    # ticks = np.linspace(-180, 180, 7)
    ticks = np.arange(-180, 181, 45)
    phase_ax.yaxis.set_ticks(ticks)
    phase_ax.semilogx(ff, np.angle(tf, True), **kwargs)
    phase_ax.set_ylabel('Phase [deg]')
    phase_ax.set_xlabel('Frequency [Hz]')
    plt.setp(mag_ax.get_xticklabels(), visible=False)
    mag_ax.grid(True, which='both', alpha=0.5)
    mag_ax.grid(True, alpha=0.25, which='minor')
    phase_ax.grid(True, which='both', alpha=0.5)
    phase_ax.grid(True, alpha=0.25, which='minor')
    if newFig:
        return fig
