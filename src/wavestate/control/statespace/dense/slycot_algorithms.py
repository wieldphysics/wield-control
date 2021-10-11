"""
"""
from __future__ import division, print_function, unicode_literals
import declarative
import numpy as np

import slycot


def rescale_slycot(A, B, C, D, E):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    out = slycot.transform.tb01id(
        n,
        m,
        p,
        10,
        np.copy(A),
        np.copy(B),
        np.copy(C),
        job = 'A',
    )
    s_norm, A, B, C, scale = out
    return declarative.Bunch(
        ABCDE = (A, B, C, D, E),
        s_norm = s_norm,
        scale = scale,
    )
