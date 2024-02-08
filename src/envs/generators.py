
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

from src.envs.diag_t import DiagT

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

    def generate(self, rng):
        qb_range = range(self.min_qubits, self.max_qubits + 1, self.qubit_step)
        dp_range = range(self.min_depth, self.max_depth + 1, self.depth_step)
        n_qubits = rng.choice(qb_range).item()
        depth = rng.choice(dp_range).item()

        circ = DiagT(n_qubits)
        t = pauliopt.pi / 4
        for _ in range(depth):
            size = rng.randint(4, min(circ.n_qubits, 8) + 1)
            qubits = rng.choice(range(circ.n_qubits), size=size, replace=False).tolist()
            times = rng.randint(1, 8)
            circ = circ.spider_nest(qubits, k=1)
        items = list(circ._dict.items())
        np.random.shuffle(items)
        inp = DiagT(n_qubits, dict(items[len(items)//3:]))
        out = DiagT(n_qubits, dict(items[:len(items)//3]))

        return inp, out, n_qubits, depth

    def evaluate(self, src, tgt, hyp):
        """
        Evaluate an example for the model.
        By construction, the source and target are always equivalent.

        Arguments
        ---------
        src: DiagT
            the generated original circuit
        tgt: DiagT
            the generated output circuit
        hyp: DiagT
            the circuit output by model

        Returns
        -------
        int
            whether the `src` is equivalent to the `hyp` (0 or 1)
        int
            (n_h) number of gadgets in hypothesis
        int
            (n_t) number of gadgets in target
        int
            (n_s) number of gadgets in source
        """
        e = src + hyp.adjoint()
        if e.is_id():
            n_s = src.n_gadgets()
            n_t = src.n_gadgets()
            n_h = src.n_gadgets()
            return 1, n_h, n_t, n_s
        return 0, 0, 0, 0
