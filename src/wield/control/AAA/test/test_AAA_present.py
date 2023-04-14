# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC0-1.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import scipy.signal
import pytest

from wavestate.iirrational.representations import asZPKTF
from wavestate.AAA import tfAAA

from wavestate.utilities.mpl import mplfigB, generate_stacked_plot_ax, asavefig, logspaced


from wavestate.pytest.fixtures import (  # noqa: F401
    tpath_join,
    plot,
    dprint,
    tpath,
    tpath_preclear,
    test_trigger,
)
from wavestate.pytest import Timer

asavefig.formats.svg.use = True


def test_AAA_view(tpath_join, tpath_preclear, dprint):
    ZPK1 = asZPKTF(
        (
            (
                -1,
                -2,
                -10,
                -2 + 10j,
                -2 - 10j,
            ),
            (
                -0.001,
                -0.1 + 0.5j,
                -0.1 - 0.5j,
                -0.1 + 5j,
                -0.1 - 5j,
                -0.1 + 50j,
                -0.1 - 50j,
                -4,
            ),
            500,
        )
    )
    dprint("ZEROS: ", ZPK1.zeros.fullplane)
    dprint("POLES: ", ZPK1.poles.fullplane)

    F_Hz = logspaced(0.5, 120, 200)
    axB = fitplot(ZPK1, F_Hz, dprint=dprint)
    axB.ax0.legend(framealpha=1)
    axB.save(tpath_join("First_demo200pts"))
    axB = fitplot(ZPK1, F_Hz, CLG=True, dprint=dprint)
    axB.save(tpath_join("First_demo200pts_CLG"))

    ZPK2 = asZPKTF(
        (
            (-20,),
            (
                -2 + 2j,
                -2 - 2j,
            ),
            10,
        )
    )
    axB = fitplot(ZPK2, F_Hz, dprint=dprint)
    axB.save(tpath_join("Second_demo200pts"))
    axB = fitplot(ZPK1, F_Hz, addZPK=ZPK2, dprint=dprint)
    axB.ax0.legend(framealpha=1, loc="upper right")
    axB.save(tpath_join("Third_demo200pts"))
    F_Hz = logspaced(0.5, 120, 50)
    axB = fitplot(ZPK1, F_Hz, addZPK=ZPK2, dprint=dprint)
    axB.ax0.legend(framealpha=1, loc="upper right")
    axB.save(tpath_join("Third_demo50pts"))
    return


def test_AAA_BNS(tpath_join, tpath_preclear, dprint):
    params = dict()
    params["m1"] = 30
    params["m2"] = 30
    params["f_min"] = 20
    params["f_max"] = 400
    params["deltaF"] = 1
    params["distance"] = 100e6
    F_Hz, hp, hc = gen_waveform(**params)

    axB = generate_stacked_plot_ax(
        ["ax0", "ax1"],
        heights_phys_in_default=3,
        heights_phys_in={},
        height_ratios={"ax0": 1, "ax1": 0.5},
        width_ratios=[1],
        xscales="log",
        xlim=None,
        wspacing=0.04,
        hspace=0.1,
    )
    hp *= 1e21
    axB.ax0.loglog(F_Hz, abs(hp))
    axB.ax1.semilogx(F_Hz, np.angle(hp, deg=True))
    axB.save(tpath_join("BNS"))

    dprint("N points", len(F_Hz))

    def fitplot(
        TF1,
        F_Hz,
        N=10,
        CLG=False,
        addZPK=None,
        res_tol=1e-4,
    ):
        timer = Timer(N)
        with timer:
            for _ in timer:
                results = tfAAA(
                    F_Hz=F_Hz,
                    xfer=TF1,
                    res_tol=res_tol,
                    # lf_eager = True,
                    # degree_max = 20,
                    # nconv = 1,
                    # nrel = 10,
                    # rtype = 'log',
                    # supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
                )
        dprint("Time: {}".format(timer))
        dprint("weights", results.wvals)
        dprint("poles", results.poles)
        dprint("zeros", results.zeros)
        dprint("gain", results.gain)

        TF2 = results(F_Hz)
        _, TF3 = scipy.signal.freqs_zpk(
            results.zeros, results.poles, results.gain, worN=F_Hz
        )
        axB = mplfigB(Nrows=2)

        axB = generate_stacked_plot_ax(
            ["ax0", "ax1"],
            heights_phys_in_default=3,
            heights_phys_in={},
            height_ratios={"ax0": 1, "ax1": 0.5},
            width_ratios=[1],
            xscales="log",
            xlim=None,
            wspacing=0.04,
            hspace=0.1,
        )

        axB.ax0.loglog(F_Hz, abs(TF1), label="Model")
        axB.ax0.semilogy(
            F_Hz, abs(TF2), label="Fit Bary. (order {})".format(results.order)
        )
        axB.ax0.semilogy(
            F_Hz, abs(TF3), label="Bary. ZPK (order {})".format(ZPKorder(results.zpk))
        )
        axB.ax1.semilogx(F_Hz, np.angle(TF1, deg=True))
        axB.ax1.semilogx(F_Hz, np.angle(TF2, deg=True))
        axB.ax1.semilogx(F_Hz, np.angle(TF3, deg=True))
        axB.ax0.set_ylabel("Magnitude")
        axB.ax1.set_ylabel("Phase [deg]")
        axB.ax_bottom.set_xlabel("Frequency")
        for idx, z in enumerate(results.supports):
            kw = dict()
            if idx == 0:
                kw["label"] = "supports"
            axB.ax0.axvline(z, lw=0.5, ls="--", color="black", **kw)
            axB.ax1.axvline(z, lw=0.5, ls="--", color="black")
        axB.ax0.set_title(
            "AAA Fit, {} Points, max rel. error {:0.0e}, fit time {}".format(
                len(F_Hz), res_tol, timer
            )
        )
        axB.ax0.legend(framealpha=1)
        return axB

    axB = fitplot(TF1=hp, F_Hz=F_Hz)
    axB.save(tpath_join("BNSfit"))

    axB = fitplot(TF1=abs(hp) ** 2, F_Hz=F_Hz)
    axB.save(tpath_join("BNSfitPow"))


def gen_waveform(**params):
    """Generate frequency-domain inspiral waveform

    Returns a tuple of (freq, h_plus^tilde, h_cross^tilde).

    The waveform is generated with the lalsimulation
    SimInspiralChooseFDWaveform() function.  Keyword arguments are
    used to update the default waveform parameters (see DEFAULT_PARAMS
    macro).  The mass parameters ('m1' and 'm2') should be specified
    in solar masses and the 'distance' parameter should be specified
    in parsecs**.  Waveform approximants may be given as string names
    (see `lalsimulation` documentation for more info).

    For example, to generate a 20/20 Msolar BBH waveform:

    >>> hp,hc = waveform.gen_waveform('m1'=20, 'm2'=20)

    **NOTE: The requirement that masses be specified in solar masses
    and distances in parsecs is different than that of the underlying
    lalsimulation method which expects mass and distance parameters to
    be in SI units.

    """
    import lalsimulation
    from inspiral_range import waveform
    from inspiral_range import const

    iparams = dict(waveform.DEFAULT_PARAMS)
    iparams.update(**params)

    # convert to SI units
    iparams["distance"] *= const.PC_SI
    iparams["m1"] *= const.MSUN_SI
    iparams["m2"] *= const.MSUN_SI
    iparams["approximant"] = lalsimulation.SimInspiralGetApproximantFromString(
        iparams["approximant"]
    )

    m = iparams["m1"] + iparams["m2"]

    # calculate delta F based on frequency of inner-most stable
    # circular orbit ("fisco")
    fisco = (const.c ** 3) / (const.G * (6 ** 1.5) * 2 * np.pi * m)
    df = 2 ** (np.max([np.floor(np.log(fisco / 4096) / np.log(2)), -6]))

    # FIXME: are these limits reasonable?
    if iparams["deltaF"] is None:
        iparams["deltaF"] = df
    # iparams['f_min'] = 0.1
    # iparams['f_max'] = 10000

    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(**iparams)

    freq = hp.f0 + np.arange(len(hp.data.data)) * hp.deltaF
    print("f0", hp.f0)
    select = abs(hp.data.data) > 0

    return freq[select], hp.data.data[select], hc.data.data[select]


def ZPKorder(ZPK):
    Z, P, K = ZPK
    return max(len(Z), len(P))


@pytest.mark.xfail(reason="Needs updated GWINC interface")
def test_AAA_GWINC(tpath_join, tpath_preclear, dprint):
    import gwinc
    import numpy as np

    F_Hz = logspaced(5, 2000, 300)
    B = gwinc.load_budget("aLIGO", freq=F_Hz)
    traces = B.run()
    fig = gwinc.plot_noise(traces)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("PyGWINC aLIGO noise budget")
    ax.set_ylabel("Strain ASD [h/rtHz]")
    fig.savefig(tpath_join("GWINC.pdf"))
    fig.savefig(tpath_join("GWINC.svg"))
    traces2 = dict()
    traces3 = dict()

    timer = Timer(3)
    with timer:
        for N in timer:
            for name, subtrace in traces.items():
                dprint("name", name)
                rescale = 1e45
                data2 = rescale * subtrace.PSD
                select = data2 > 1e-6
                results = tfAAA(
                    F_Hz=F_Hz[select],
                    xfer=data2[select],
                    res_tol=1e-4,
                    # lf_eager = True,
                    # degree_max = 20,
                    # nconv = 1,
                    # nrel = 10,
                    # rtype = 'log',
                    # supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
                )

                # TF2 = results(F_Hz)
                _, TF3 = scipy.signal.freqs_zpk(
                    results.zeros, results.poles, results.gain, worN=F_Hz
                )
                label = subtrace.style.get("label", name)
                other2 = dict(other)
                other2["label"] = label + " (order {})".format(ZPKorder(results.zpk))
                if name == "Total":
                    traces2[name] = (TF3 / rescale, other)
                    traces3[name] = (TF3 / rescale, other2)
                else:
                    traces2[name] = (TF3 / rescale, other2)
                    traces3[name] = (TF3 / rescale, other2)

    fig = gwinc.plot_noise(F_Hz, traces2)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(
        "PyGWINC aLIGO noise budget\nAAA Fit, {} Points, max rel. error {:0.0e}, fit time {}".format(
            len(F_Hz), 1e-4, timer
        )
    )
    ax.set_ylabel("Strain ASD [h/rtHz]")
    fig.savefig(tpath_join("GWINCfit.pdf"))
    fig.savefig(tpath_join("GWINCfit.svg"))

    fig = gwinc.plot_noise(F_Hz, traces3)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(
        "PyGWINC aLIGO noise budget\nAAA Fit, {} Points, max rel. error {:0.0e}, fit time {}".format(
            len(F_Hz), 1e-4, timer
        )
    )
    ax.set_ylabel("Strain ASD [h/rtHz]")
    fig.savefig(tpath_join("GWINCfitUgly.pdf"))
    fig.savefig(tpath_join("GWINCfitUgly.svg"))

    # axB = fitplot(TF1 = hp, F_Hz = F_Hz)
    # axB.save(tpath_join('BNSfit'))

    # axB = fitplot(TF1 = abs(hp)**2, F_Hz = F_Hz)
    # axB.save(tpath_join('BNSfitPow'))


def fitplot(
    ZPK,
    F_Hz,
    N=10,
    CLG=False,
    addZPK=None,
    res_tol=1e-4,
    dprint=print,
):
    TF1 = ZPK.xfer_eval(F_Hz=F_Hz)
    order = ZPK.order
    if CLG:
        TF1 = 1 / (1 - TF1)
    if addZPK:
        TF1 = TF1 + addZPK.xfer_eval(F_Hz=F_Hz)
        order = ZPK.order + addZPK.order
    timer = Timer(N)

    with timer:
        for _ in timer:
            results = tfAAA(
                F_Hz=F_Hz,
                xfer=TF1,
                res_tol=res_tol,
                # lf_eager = True,
                # degree_max = 20,
                # nconv = 1,
                # nrel = 10,
                # rtype = 'log',
                # supports = (1e-2, 1e-1, 4.2e-1, 5.5e-1, 1.5, 2.8, 1, 5e-1, 2),
            )
    dprint("Time: {}".format(timer))
    dprint("weights", results.wvals)
    dprint("poles", results.poles)
    dprint("zeros", results.zeros)
    dprint("gain", results.gain)
    dprint("poles1", ZPK.poles.fullplane)
    dprint("zeros1", ZPK.zeros.fullplane)
    dprint("order1", ZPK.order, len(ZPK.zeros), len(ZPK.poles))
    dprint("order2", results.order)

    TF2 = results(F_Hz)
    _, TF3 = scipy.signal.freqs_zpk(
        results.zeros, results.poles, results.gain, worN=F_Hz
    )
    axB = mplfigB(Nrows=2)

    axB = generate_stacked_plot_ax(
        ["ax0", "ax1"],
        heights_phys_in_default=3,
        heights_phys_in={},
        height_ratios={"ax0": 1, "ax1": 0.5},
        width_ratios=[1],
        xscales="log",
        xlim=None,
        wspacing=0.04,
        hspace=0.1,
    )

    axB.ax0.loglog(F_Hz, abs(TF1), label="Model (order {})".format(order))
    axB.ax0.semilogy(F_Hz, abs(TF2), label="Fit Bary. (order {})".format(results.order))
    axB.ax0.semilogy(
        F_Hz, abs(TF3), label="Bary. ZPK (order {})".format(ZPKorder(results.zpk))
    )
    axB.ax1.semilogx(F_Hz, np.angle(TF1, deg=True))
    axB.ax1.semilogx(F_Hz, np.angle(TF2, deg=True))
    axB.ax1.semilogx(F_Hz, np.angle(TF3, deg=True))
    axB.ax0.set_ylabel("Magnitude")
    axB.ax1.set_ylabel("Phase [deg]")
    axB.ax_bottom.set_xlabel("Frequency")
    for idx, z in enumerate(results.supports):
        kw = dict()
        if idx == 0:
            kw["label"] = "supports"
        axB.ax0.axvline(z, lw=0.5, ls="--", color="black", **kw)
        axB.ax1.axvline(z, lw=0.5, ls="--", color="black")
    axB.ax0.set_title(
        "AAA Fit, {} Points, max rel. error {:0.0e}, fit time {}".format(
            len(F_Hz), res_tol, timer
        )
    )
    axB.ax0.legend(framealpha=1)
    return axB
