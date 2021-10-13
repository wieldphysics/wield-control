"""
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy
import scipy.signal

from wavestate.utilities.np import logspaced
from wavestate.utilities.mpl import mplfigB
from transient.statespace import dense
from transient.statespace import ACE
from transient.statespace.dense.zpk_algorithms import zpk_cascade
from transient.statespace.dense.xfer_algorithms import ss2xfer

from wavestate.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot, fpath_join,
)

import scipy.signal

c_m_s = 299792458

def print_ssd(ssd):
    print("Bnz", 1*(ssd.B != 0))
    print("Anz", 1*(ssd.A != 0))
    print("E", ssd.E)
    print("C", 1*(ssd.C != 0))
    print("D", ssd.D)


def test_ACE_tupleize(pprint, tpath_join, fpath_join):
    pprint(ACE.tupleize('A.B'))
    pprint(ACE.tupleize(('A.B',)))
    pprint(ACE.tupleize((None, ('A.B',), 'C')))
    pprint(ACE.tupleize(None))


def test_xfers_ACE(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    Zc = [-1 + 1j, -1 + 5j]
    Zr = [-100, -200]
    Pc = [-1 + 2j, -1 + 6j]
    Pr = [-10, -20]
    #Zc = []
    #Zr = []
    k = 1

    Zc = 2 * np.pi * np.asarray(Zc)
    Zr = 2 * np.pi * np.asarray(Zr)
    Pc = 2 * np.pi * np.asarray(Pc)
    Pr = 2 * np.pi * np.asarray(Pr)

    Z = np.concatenate([Zc, Zc.conjugate(), Zr])
    P = np.concatenate([Pc, Pc.conjugate(), Pr])

    sys1 = dense.ZPKdict(
        name = 'zpk',
        zdict = dict(c = Zc, r = Zr),
        pdict = dict(c = Pc, r = Pr),
        k = k,
    )
    printSSBnz(sys1)

    F_Hz = logspaced(.01, 1e3, 1000)
    ABCD_xfer = sys1.xfer(
        F_Hz = F_Hz,
        iname = 'zpk.i0',
        oname = 'zpk.o0',
    )

    #TODO, use IIRrational version since statespace is likely more numerically
    #stable than crappy scipy implementation
    w, zpk_zfer = scipy.signal.freqs_zpk(Z, P, k, 2 * np.pi * F_Hz)

    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(ABCD_xfer))
        axB.ax1.semilogx(F_Hz, np.angle(ABCD_xfer, deg = True))
        axB.ax0.loglog(F_Hz, abs(zpk_zfer))
        axB.ax1.semilogx(F_Hz, np.angle(zpk_zfer, deg = True))
        axB.save(tpath_join('test'))
    with test_trigger(trigger, plot = plot):
        np.testing.assert_allclose(ABCD_xfer, zpk_zfer, rtol = 1e-5)

    #OK, now begin ACE testing
    ace1 = ACE.ACE.from_ABCD(sys1.A, sys1.B, sys1.C, sys1.D, sys1.E)
    ace1.io_add('in', {'I' : None}, constr = True)
    ssB0 = ace1.statespace(inputs = ['in'], outputs = ['O'], Dreduce = False)

    ABCDs = zpk_cascade(zr = Zr, zc = Zc, pr = Pr, pc = Pc, k = k)
    syslist = []
    for (A, B, C, D, E) in ABCDs:
        syslist.append(ACE.ACE.from_ABCD(A, B, C, D, E))

    ace2 = ACE.ACE()
    for idx, sys in enumerate(syslist):
        ace2.insert(sys, cmn = "sys{}".format(idx))
        pprint(sys.Eranks)
    for idx in range(len(syslist) - 1):
        ace2.bind_equal({"sys{}.O".format(idx), "sys{}.I".format(idx + 1)}, constr = 's{}{}'.format(idx, idx+1))
    ace2.io_add('in', {'sys0.I' : None}, constr = True)
    pprint("TEST", ace2.cr2stA)

    pprint(ace2.states_edges())
    pprint(ace2.states_reducible())
    pprint(ace2.io2st)

    ssB1 = ace2.statespace(inputs = ['in'], outputs = ['sys2.O'], Dreduce = False)

    #no longer a good test, since there isn't an internal constraint
    sccs = ace2.states_reducible_sccs()
    pprint("TEST", ace2.cr2stA)
    pprint(sccs)
    pprint('scc2', sccs)
    for st_set, cr_set in sccs[:]:
        pprint("SCC_reduce", st_set, cr_set)
        ace2.simplify_scc(st_set, cr_set)
    #pprint(ace2.io2st)
    pprint("TEST", ace2.cr2stA)

    #pprint(ssB1)
    ssB2 = ace2.statespace(inputs = ['in'], outputs = ['sys2.O'], Dreduce = False)
    ssB3 = ace2.statespace(inputs = ['in'], outputs = ['sys2.O'], Dreduce = True)
    printSSBnz(ssB1)
    print('----------------------------')
    printSSBnz(ssB2)
    print('----------------------------')
    printSSBnz(ssB3)
    #pprint(ace2.states_edges())
    #pprint(ace2.st2crE)
    #pprint(ace2.states_reducible())

    xfer = ss2xfer(*ssB0.ABCDE, F_Hz = F_Hz)
    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(xfer))
        axB.ax1.semilogx(F_Hz, np.angle(xfer, deg = True))
        axB.ax0.loglog(F_Hz, abs(ABCD_xfer))
        axB.ax1.semilogx(F_Hz, np.angle(ABCD_xfer, deg = True))
        axB.save(tpath_join('test_ssB0'))
    with test_trigger(trigger, plot = plot):
        np.testing.assert_allclose(xfer, ABCD_xfer, rtol = 1e-5)

    xfer1 = ss2xfer(ssB1.A, ssB1.B, ssB1.C, ssB1.D, ssB1.E, F_Hz = F_Hz)
    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(xfer1))
        axB.ax1.semilogx(F_Hz, np.angle(xfer1, deg = True))
        axB.ax0.loglog(F_Hz, abs(ABCD_xfer))
        axB.ax1.semilogx(F_Hz, np.angle(ABCD_xfer, deg = True))
        axB.save(tpath_join('test_ssB1'))
    with test_trigger(trigger, plot = plot):
        np.testing.assert_allclose(xfer1, ABCD_xfer, rtol = 1e-5)

    xfer2 = ss2xfer(*ssB2.ABCDE, F_Hz = F_Hz)
    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(xfer2))
        axB.ax1.semilogx(F_Hz, np.angle(xfer2, deg = True))
        axB.ax0.loglog(F_Hz, abs(ABCD_xfer))
        axB.ax1.semilogx(F_Hz, np.angle(ABCD_xfer, deg = True))
        axB.save(tpath_join('test_ssB2'))
    with test_trigger(trigger, plot = plot):
        np.testing.assert_allclose(xfer2, ABCD_xfer, rtol = 1e-5)

    xfer3 = ss2xfer(*ssB3.ABCDE, F_Hz = F_Hz)
    def trigger(fail, plot):
        axB = mplfigB(Nrows = 2)
        axB.ax0.loglog(F_Hz, abs(xfer3))
        axB.ax1.semilogx(F_Hz, np.angle(xfer3, deg = True))
        axB.ax0.loglog(F_Hz, abs(ABCD_xfer))
        axB.ax1.semilogx(F_Hz, np.angle(ABCD_xfer, deg = True))
        axB.save(tpath_join('test_ssB3'))
    with test_trigger(trigger, plot = plot):
        np.testing.assert_allclose(xfer3, ABCD_xfer, rtol = 1e-5)
    return


def nz(M):
    return 1*(M != 0)


def printSSBnz(ssb):
    Astr = np.array2string(nz(ssb.A))
    Bstr = np.array2string(nz(ssb.B))
    Cstr = np.array2string(nz(ssb.C))
    Dstr = np.array2string(nz(ssb.D))
    ziplines(Astr + '\n\n' + Cstr, Bstr + '\n\n' + Dstr, delim = ' | ')


def ziplines(*args, delim = ''):
    widths = []
    for arg in args:
        w = max(len(line) for line in arg.splitlines())
        widths.append(w)
    for al in zip(*[arg.splitlines() for arg in args]):
        line = []
        for a, w in zip(al, widths):
            line.append(a + ' '*(w - len(a)))
        print(delim.join(line))

def test_reducer(pprint, test_trigger, tpath_join, tpath_preclear, plot):
    ace = ACE.ACE()

    seis_acc = dense.zpk_rc(
        name = 'seis_acc',
        Zc = [-0.4 - 0.200j],
        Zr = [],
        Pc = [-0.10 - 0.200j],
        Pr = [],
        k = 1e-6,
        convention = 'scipyHz',
    )
    Aseis_acc = ACE.ACE.from_ABCD(*seis_acc.ABCDE)
    ace.insert(Aseis_acc, cmn = "g1")

    seis1_sensing = dense.zpk_rc(
        name = 'seis1',
        Zc = [],
        Zr = [-.100, -.100],
        Pc = [],
        Pr = [-.001, -.001],
        k = 1e-9,
        convention = 'scipyHz',
    )
    Aseis1_sensing = ACE.ACE.from_ABCD(*seis1_sensing.ABCDE)
    ace.insert(Aseis1_sensing, cmn = "s1")
    seis2_sensing = dense.zpk_rc(
        name = 'seis2',
        Zc = [],
        Zr = [],
        Pc = [],
        Pr = [],
        k = 1e-7,
        convention = 'scipyHz',
    )
    Aseis2_sensing = ACE.ACE.from_ABCD(*seis2_sensing.ABCDE)
    ace.insert(Aseis2_sensing, cmn = "s2")

    ace.bind_equal({'g1.O', 's2.O'})
    ace.bind_equal({'g1.O', 's1.O'})
    ace.io_input('g1.I')

    sccs = ace.states_reducible_sccs()
    pprint("TEST", ace.cr2stA)
    pprint(sccs)
    pprint('scc2', sccs)
    for st_set, cr_set in sccs[:]:
        pprint("SCC_reduce", st_set, cr_set)
        ace.simplify_scc(st_set, cr_set)
    return
