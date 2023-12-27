"""
"""
import numpy as np
import matplotlib.pyplot as plt
import control
from scipy.signal import cheby2
from wield.control import SISO
from wield.utilities.mpl import mplfigB

from wield.pytest import tjoin, fjoin, dprint  # noqa

from wield.control.utilities import algorithm_choice


def test_algorithm_choices():
    """
    This is a test of the numerical stability of a very large ZPK filter that spans 18 orders of magnitude

    """
    zpk2ss_ranks = algorithm_choice.algorithm_choices_defaults['zpk2ss']
    dprint(zpk2ss_ranks)
    for k, r in zpk2ss_ranks.items():
        algorithm_choices = {'zpk2ss': {k: 1000}}
        print("Rank top: ", k)

        filt = SISO.zpk(
            [], [], 1,
            algorithm_choices=algorithm_choices
        )
        filtss = filt.asSS

        dprint(filt.algorithm_choices)
        dprint(filt.algorithm_ranking)

        func = algorithm_choice.algorithm_mappings_name2func[k]
        assert (filt.algorithm_ranking['zpk2ss'][0] == func)
        assert (filtss.algorithm_ranking['zpk2ss'][0] == func)


    return


