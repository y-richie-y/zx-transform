""" Some example code for spider nests """

from typing import OrderedDict

from itertools import combinations
import pytest

import numpy as np

from src.envs.diag_t import DiagT


def is_id(circ):
    """ Checks if circuit is identity by converting to multi-linear form """

    # only works for angle8
    d = OrderedDict()
    for legs, angle in circ._dict.items():
        for i in range(1, len(legs) + 1):
            for qs in combinations(legs, i):
                if qs not in d:
                    d[qs] = 0
                coeff = -1 * (-2) ** i
                d[qs] += float(angle) * coeff / 16
                # TODO check why 16 works, not 8
    return all(abs(angle) % 1 < 1e-5 for angle in d.values())


@pytest.mark.parametrize("n_qubits", range(4, 10))
def test_equality(n_qubits):
    # generate random circuits and check if they are equal to identit
    # n_qubits = 10
    circ = DiagT(n_qubits, ignore_cliffords=False)
    rng = np.random

    for _ in range(20):
        circ.random_walk(circ, rng, niter=100)
        assert is_id(circ)
