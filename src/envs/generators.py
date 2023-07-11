
from abc import ABC, abstractmethod
import numpy as np
import math
import pyzx
from logging import getLogger
from pyzx.generate import cliffords, cnots, cliffordT

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
