import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy

from transient.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot, fpath_join,
)
from transient.utilities.mpl import mplfigB
import transient
import re

from transient.SFLU import SFLU

def T_SFLU_pre1(pprint, tpath_join, fpath_join):
    """
    Setup 2-node problem, nodes a and b with edges Eaa, Eab, Eba, Ebb.
    """

    edges = {
        ('a', 'a_in') : 1,
        ('a_out', 'a') : 1,
        ('a', 'a') : 'Eaa',
        ('b', 'a') : 'Eba',
        ('a', 'b') : 'Eab',
        ('b', 'b') : 'Ebb',
    }
    sflu = SFLU.SFLU(edges)
    
    print('-----------------------')
    pprint(sflu.edges)
    sflu.reduce('a')
    print('-----------------------')
    pprint(sflu.edges)
    pprint(sflu.row2)
    pprint(sflu.col2)
    sflu.reduce('b')
    print('-----------------------')
    pprint(sflu.edges)
    pprint(sflu.row2)
    pprint(sflu.col2)
    print('-----------------------oplist')
    pprint(sflu.oplistE)
    pprint(sflu.subinverse('a_out', 'a_in'))

    return

def T_sympy(pprint):
    x = sympy.Symbol('x', commutative = False)
    y = sympy.Symbol('y', commutative = False)

    def symbols():
        i = 0
        while True:
            yield sympy.Symbol('S_{}'.format(i), commutative = False)
            i += 1
    E = x * y**2 - (y**-2 * x)**-2
    pprint(E)
    subexprs, cse_expr = sympy.cse(E, symbols = symbols())
    def transform(ex):
        return re.sub(r'([^*])\*([^*])', r'\1 @ \2', str(ex), count = 0)
    for expr in subexprs:
        print(str(expr[0]), '=', transform(expr[1]))
    print(transform(cse_expr[0]))
