
from abc import ABC, abstractmethod
import random
import numpy as np
import math
import pyzx
from logging import getLogger
from pyzx.generate import cliffords, cnots, cliffordT

from pauliopt import phase as pauliopt
from itertools import combinations
from collections import OrderedDict

logger = getLogger()


class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Graphs(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.min_qubits = params.min_qubits
        self.max_qubits = params.max_qubits
        self.qubit_step = params.qubit_step
        self.min_depth = params.min_depth
        self.max_depth = params.max_depth
        self.depth_step = params.depth_step
        self.circuit_type = params.circuit_type
            
    def generate(self, rng):

        qubits = rng.choice(range(self.min_qubits, self.max_qubits + 1, self.qubit_step))
        depth = rng.choice(range(self.min_depth, self.max_depth + 1, self.depth_step))
        if self.circuit_type == "clifford":
            circ = cliffords(qubits, depth)
        elif self.circuit_type == "cnot":
            circ = cnots(qubits, depth)
        else:
            circ = cliffordT(qubits, depth)
        pyzx.to_gh(circ)
        
        simp = circ.copy()
        pyzx.full_reduce(simp)
        return circ, simp, qubits, depth

    def evaluate(self, src, tgt, hyp):
        e = src + hyp.adjoint()
        pyzx.full_reduce(e)
        if e.is_id():
            nodes_s = src.num_vertices()
            nodes_t = tgt.num_vertices()
            nodes_h = hyp.num_vertices()
            return 1, nodes_h, nodes_t, nodes_s
        return 0,0,0,0


class Circuits(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.min_qubits = params.min_qubits
        self.max_qubits = params.max_qubits
        self.qubit_step = params.qubit_step
        self.min_depth = params.min_depth
        self.max_depth = params.max_depth
        self.depth_step = params.depth_step
        self.walk_steps = params.walk_steps

    def random_walk(self, circ, rng, niter=100):
        """ Random walk on the space of circuits using spider nest identity """
        for _ in range(niter):
            size = rng.randint(4, min(circ.num_qubits, 8))
            qubits = rng.choice(range(circ.num_qubits), size=size).tolist()
            circ = self.spider_nest(circ, qubits)
        return circ

    def spider_nest(self, circ, qubits):
        """
        Apply the n-qubit spider nest identity to the selected qubits.
        """

        n = len(qubits)

        if n > circ.num_qubits or n < 4:
            raise ValueError(f"Invalid number of qubits {n}")
        for q in qubits:
            angle = (n - 2) * (n - 3) * pauliopt.pi / 8
            circ >>= pauliopt.Z(angle) @ {q}
        for q0, q1 in combinations(qubits, 2):
            angle = - (n - 3) * pauliopt.pi / 4
            circ >>= pauliopt.Z(angle) @ {q0, q1}
        for q0, q1, q2 in combinations(qubits, 3):
            angle = pauliopt.pi / 4
            circ >>= pauliopt.Z(angle) @ {q0, q1, q2}
        angle = - pauliopt.pi / 4
        circ >>= pauliopt.Z(angle) @ qubits

        return circ

    def compress(self, circ):
        d = OrderedDict()
        for gadget in circ.gadgets:
            basis = gadget.basis
            legs = tuple(sorted(gadget.qubits))
            angle = gadget.angle
            if (basis, legs) not in d:
                d[(basis, legs)] = angle - angle
            d[(basis, legs)] += angle

        new_circ = pauliopt.PhaseCircuit(circ.num_qubits)
        for (basis, legs), angle in d.items():
            if angle == 0:
                continue
            if basis == "Z":
                new_circ >>= pauliopt.Z(angle) @ legs
            elif basis == "X":
                new_circ >>= pauliopt.X(angle) @ legs
            else:
                raise ValueError(f"Unknown basis {basis}")

        return new_circ

    def is_id(circ):
        """ Checks circuit is identity by converting to multi-linear form """
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

    def generate(self, rng):
        qubits = rng.choice(range(self.min_qubits, self.max_qubits + 1, self.qubit_step)).item()
        depth = rng.choice(range(self.min_depth, self.max_depth + 1, self.depth_step)).item()

        circ = pauliopt.PhaseCircuit(qubits)
        t = pauliopt.pi / 4
        for _ in range(depth):
            phase = t * rng.choice(range(8))
            n_legs = rng.choice(range(1, max(4, qubits)))
            legs = rng.choice(range(qubits), size=n_legs, replace=False).tolist()
            circ >>= pauliopt.Z(phase) @ legs
        self.compress(circ)
        orig_circ = circ.cloned()
        self.random_walk(circ, rng, niter=self.walk_steps)
        self.compress(circ)

        # backwards generation
        return circ, orig_circ, qubits, depth

    def evaluate(self, src, tgt, hyp):
        e = src + hyp.adjoint()
        for g in hyp.gadgets[::-1]:
            src >>= pauliopt.Z(-g.angle) @ g.qubits
        if self.is_id(e):
            n_s = len(src)
            n_t = len(tgt)
            n_h = len(hyp)
            return 1, n_h, n_t, n_s
        return 0, 0, 0, 0
