""" Some example code for spider nests """

import random
from typing import OrderedDict
from pauliopt.phase import PhaseCircuit, Z, X, pi

from itertools import combinations
import math

circ = PhaseCircuit(5)
circ >>= Z(3 * pi/4) @ {0, 1, 2, 3, 4}
circ >>= Z(-pi/4) @ {0, 1, 4}
circ >>= Z(2 * pi/4) @ {2, 3, 4}
circ >>= Z(pi/2) @ {0, 1, 2, 3, 4}

def encode(circ):
    tokens = [f"nqubits={circ.num_qubits}"]
    for gadget in circ.gadgets:
        angle8 = int(float(gadget.angle) * 4 / float(pi))
        tokens.extend([f"q{i}" for i in gadget.qubits])
        tokens.append(f"{gadget.basis}({angle8}T)")
    return tokens


def decode(tokens):
    num_qubits = int(tokens[0].split("=")[1])
    circ = PhaseCircuit(num_qubits)
    legs = set()
    for token in tokens[1:]:
        if token.startswith("q"):
            legs.add(int(token[1:]))
        else:
            basis = token[0]
            angle4 = int(token[2:-2])
            angle = angle4 * pi / 4
            if basis == "Z":
                circ >>= Z(angle) @ legs
            elif basis == "X":
                circ >>= X(angle) @ legs
            else:
                raise ValueError(f"Unknown basis {basis}")
            legs = set()

    return circ


def compress(circ):
    d = OrderedDict()
    for gadget in circ.gadgets:
        basis = gadget.basis
        legs = tuple(sorted(gadget.qubits))
        angle = gadget.angle
        if (basis, legs) not in d:
            d[(basis, legs)] = angle - angle
        d[(basis, legs)] += angle
    
    new_circ = PhaseCircuit(circ.num_qubits)
    for (basis, legs), angle in d.items():
        if angle == 0:
            continue
        if basis == "Z":
            new_circ >>= Z(angle) @ legs
        elif basis == "X":
            new_circ >>= X(angle) @ legs
        else:
            raise ValueError(f"Unknown basis {basis}")
    return new_circ


def spider_nest(circ, qubits):
    """
    Apply the n-qubit spider nest identity to the selected qubits.
    """

    n = len(qubits)

    if n > circ.num_qubits or n < 4:
        raise ValueError(f"Invalid number of qubits {n}")
    for q in qubits:
        angle = (n - 2) * (n - 3) * pi / 8
        circ >>= Z(angle) @ {q}
    for q0, q1 in combinations(qubits, 2):
        angle = - (n - 3) * pi / 4
        circ >>= Z(angle) @ {q0, q1}
    for q0, q1, q2 in combinations(qubits, 3):
        angle = pi / 4
        circ >>= Z(angle) @ {q0, q1, q2}
    angle = - pi / 4
    circ >>= Z(angle) @ qubits

    return (circ)


def is_id(circ):
    """ Checks if circuit is identity by converting to multi-linear form """

    d1 = OrderedDict()
    for gadget in circ.gadgets:
        basis = gadget.basis
        legs = tuple(sorted(gadget.qubits))
        angle = gadget.angle
        if basis != "Z":
            raise ValueError(f"Invalid basis {basis}")
        if legs not in d1:
            d1[legs] = 0
        d1[legs] += float(angle) / 2 / math.pi

    d2 = OrderedDict()
    for legs, angle in d1.items():
        for i in range(1, len(legs) + 1):
            for qs in combinations(legs, i):
                if qs not in d2:
                    d2[qs] = 0
                coeff = -1 * (-2) ** i
                d2[qs] += float(angle) * coeff
    return all(abs(angle) % 1 < 1e-5 for angle in d2.values())

def random_walk(circ, niter=100):
    """ Random walk on the space of circuits using spider nest identity """
    for _ in range(niter):
        size = random.randint(4, min(circ.num_qubits, 8))
        qubits = sorted(random.sample(range(circ.num_qubits), size))
        circ = spider_nest(circ, qubits)
    
    return circ


decode(encode(circ))

circ = PhaseCircuit(9)
spider_nest(circ, [0,1,2,3])

random_walk(circ, niter=10)
is_id(circ)
