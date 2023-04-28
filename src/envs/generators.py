
from abc import ABC, abstractmethod
import numpy as np
import math
import pyzx
from logging import getLogger
from pyzx.generate import cliffords, cnots

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
        self.min_depth = params.min_depth
        self.max_depth = params.max_depth
            
    def generate(self, rng):
        qubits = rng.randint(self.min_qubits, self.max_qubits + 1)
        depth = rng.randint(self.min_depth, self.max_depth + 1)
        circ = cliffords(qubits, depth)
        circ = pyzx.to_gh(circ)

        simp = circ.copy()
        pyzx.full_reduce(simp)
        return circ, simp

    def evaluate(self, src, tgt, hyp):
        t = hyp.verify_equality(src)
        if t:
            return 0, 0, 0, 0
        return -1,-1,-1,-1
