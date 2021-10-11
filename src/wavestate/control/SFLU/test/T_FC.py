import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy

from transient.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot, fpath_join,
)
from transient.utilities.mpl import mplfigB
from transient.utilities.np import logspaced
import transient
import re

from transient.statespace import tupleize
from transient.SFLU import SFLU
from quantum_lib import mats_planewave as ilib

def T_SFLU_FP(pprint, tpath_join, fpath_join):
    """
    Setup a cavity with fields
    a2:a1 ---- b1:b2
    """

    edges = {
        ('a2_o', 'a2_i') : 'r_a2',  # make this a load operator
        ('a1_o', 'a2_i') : 't_a',
        ('a1_o', 'a1_i') : 'r_a',
        ('a2_o', 'a1_i') : 't_a',
        ('b1_o', 'b1_i') : 'r_b',
        ('b2_o', 'b1_i') : 't_b',
        ('b2_o', 'b2_i') : 'r_b2',  # make this a load operator
        ('b1_o', 'b2_i') : 't_b',
        ('b1_i', 'a1_o') : 'Lmat',
        ('a1_i', 'b1_o') : 'Lmat',
    }
    sflu = SFLU.SFLU(edges)
    print('inputs', sflu.inputs)
    print('outputs', sflu.outputs)
    
    print('row1', sflu.row)
    print('row2', sflu.row2)
    print('col1', sflu.col)
    print('col2', sflu.col2)
    print('-----------------------')
    sflu.reduce('a1_i')
    sflu.reduce('b1_i')
    sflu.reduce('b1_o')
    sflu.reduce('a1_o')
    #sflu.reduce('a2_i')
    #sflu.reduce('a2_o')
    #sflu.reduce('b2_i')
    #sflu.reduce('b2_o')
    print('-----------------------')

    print('row1', sflu.row)
    print('row2', sflu.row2)
    print('col1', sflu.col)
    print('col2', sflu.col2)
    pprint(sflu.oplistE)
    oplistN = sflu.subinverse('a2_o', 'a2_i')
    pprint(oplistN)

    F_Hz = logspaced(1, 1e5, 1000)
    L_m  = 4000
    i2pi = 2j*np.pi
    R_a = .01
    c_m_s = 3e8

    Espace = computeLU(
        oplistE = sflu.oplistE,
        edges = sflu.edgesO,
        nodesizes = {},
        defaultsize = 2,
        **dict(
            r_a  = ilib.Id * R_a**0.5,
            r_a2 = ilib.Id * -R_a**0.5,
            t_a  = ilib.Id * (1 - R_a)**0.5,
            r_b  = ilib.Id * 1,
            t_b  = ilib.Id * 0,
            r_b2 = ilib.Id * -1,
            #Lmat = ilib.Mrotation(2 * np.pi * F_Hz * c_m_s / L_m),
            Lmat = ilib.diag(np.exp(-i2pi * F_Hz * L_m / c_m_s)),
        ),
    )
    print(list(Espace.keys()))

    SI = compute_subinverse(
        oplistN = oplistN,
        Espace = Espace,
        nodesizes = {},
        defaultsize = 2,
    )
    axB = mplfigB(Nrows = 2)
    axB.ax0.loglog(F_Hz, abs(SI[:, 0, 0]))
    axB.ax1.semilogx(F_Hz, np.angle(SI[:, 0, 0]))
    axB.save(tpath_join('reflSI'))
    #print(Espace)
    return


def computeLU(
    oplistE,
    edges,
    nodesizes = {},
    defaultsize = None,
    **kwargs,
):
    Espace = dict()

    #load all of the initial values
    #TODO, allow this to include operations
    for ek, ev in edges.items():
        r, c = ek
        r = tupleize.tupleize(r)
        c = tupleize.tupleize(c)
        Espace[tupleize.EdgeTuple(r, c)] = kwargs[ev]

    for op in oplistE:
        print(op)
        if op.op == 'E_CLG':
            arg, = op.args
            E = Espace[arg]
            E.shape[-1]
            assert(E.shape[-1] == E.shape[-2])
            I = np.eye(E.shape[-1])
            I = I.reshape((1,) * (len(E.shape) - 2) + (E.shape[-2:]))

            E2 = np.linalg.inv(I - E)

            Espace[op.targ] = E2

        elif op.op == 'E_CLGd':
            arg, = op.args
            size = nodesizes.get(arg, defaultsize)
            I = np.eye(size)
            Espace[op.targ] = I

        elif op.op == 'E_mul2':
            arg1, arg2 = op.args
            E1 = Espace[arg1]
            E2 = Espace[arg2]
            Espace[op.targ] = E1 @ E2

        elif op.op == 'E_mul3':
            arg1, arg2, arg3 = op.args
            E1 = Espace[arg1]
            E2 = Espace[arg2]
            E3 = Espace[arg3]
            Espace[op.targ] = E1 @ E2 @ E3

        elif op.op == 'E_mul3add':
            arg1, arg2, arg3, argA = op.args
            E1 = Espace[arg1]
            E2 = Espace[arg2]
            E3 = Espace[arg3]
            EA = Espace[argA]
            Espace[op.targ] = E1 @ E2 @ E3 + EA

        elif op.op == 'E_assign':
            arg, = op.args
            Espace[op.targ] = Espace[arg]

        elif op.op == 'E_del':
            pass

        else:
            raise RuntimeError("Unrecognized Op {}".format(op))

    return Espace


def compute_subinverse(
    oplistN,
    Espace,
    nodesizes = {},
    defaultsize = None,
):
    Nspace = dict()

    for op in oplistN:
        if op.op == 'N_edge':
            #load an edge into a node
            Earg, = op.args
            Ntarg = Nspace.get(op.targ, None)
            E = Espace[Earg]

            if Ntarg is None:
                Nspace[op.targ] = E.copy()
            elif Ntarg.shape == E.shape:
                Nspace[op.targ] += E
            else:
                Nspace[op.targ] = Ntarg + E

        elif op.op == 'N_sum':
            argE, argN = op.args
            Ntarg = Nspace.get(op.targ, None)
            prod = Espace[argE] @ Nspace[argN]
            if Ntarg is None:
                Nspace[op.targ] = prod
            elif Ntarg.shape == prod.shape:
                Nspace[op.targ] += Espace[argE] @ Nspace[argN]
            else:
                Nspace[op.targ] = Ntarg + Espace[argE] @ Nspace[argN]

        elif op.op == 'N_ret':
            return Nspace[op.targ]

        else:
            raise RuntimeError("Unrecognized Op {}".format(op))












