"""
Representation of Diagonal CNOT-Dihedral circuits
"""

from itertools import combinations


def rev_len_sort(x):
    return -len(x), x


class DiagT:
    def __init__(self, n_qubits, _dict=None, ignore_cliffords=False):
        if _dict is None:
            _dict = dict()
        self._dict = _dict
        self.n_qubits = n_qubits
        self.ignore_cliffords = ignore_cliffords
        self.base = 2 if ignore_cliffords else 8

    def __repr__(self):
        return f"DiagT({self.n_qubits}, {self._dict}, {self.ignore_cliffords})"

    def _repr_svg_(self):
        return self.to_pauliopt()._repr_svg_()

    def add_gadget(self, phase8, qubits):
        if len(set(qubits)) != len(qubits):
            raise Exception(f"Qubits {qubits} must be unique")
        if any(q < 0 or self.n_qubits <= q for q in qubits):
            raise Exception(f"Qubits {qubits} out of range")
        if int(phase8) != phase8:
            raise Exception(f"phase8 {phase8} should be an integer")
        qubits = tuple(sorted(qubits))
        self._dict[qubits] = (phase8 + self._dict.get(qubits, 0)) % self.base
        if self._dict[qubits] == 0:
            del self._dict[qubits]
        return self

    def spider_nest(self, qubits, k=1):
        """
        Apply the n-qubit spider nest identity k times to the selected qubits.
        """
        n = len(qubits)
        k = (k + self.base) % self.base  # num of times the move is applied

        if n > self.n_qubits or n < 4:
            raise ValueError(f"Invalid number of qubits {n}")
        for q in qubits:
            angle8 = int((n - 2) * (n - 3) / 2)  # (n - 2) * (n - 3) * pi / 8
            self.add_gadget(k * angle8, (q,))
        for qs in combinations(qubits, 2):
            angle8 = - (n - 3)  # - (n - 3) * pi / 4
            self.add_gadget(k * angle8, qs)
        for qs in combinations(qubits, 3):
            angle8 = 1  # -pi / 4
            self.add_gadget(k * angle8, qs)
        angle8 = - 1  # -pi / 4
        self.add_gadget(k * angle8, qubits)
        return self

    def random_walk(self, circ, rng, niter=100):
        """ Random walk on the space of circuits using spider nest identity """
        for _ in range(niter):
            size = rng.randint(4, min(circ.n_qubits, 8) + 1)
            qubits = rng.choice(range(circ.n_qubits), size=size, replace=False).tolist()
            k = rng.choice(range(1, circ.base))
            circ.spider_nest(qubits, k)
        return circ

    def to_pauliopt(self):
        from pauliopt import phase as pauliopt
        circ = pauliopt.PhaseCircuit(self.n_qubits)
        for k in sorted(self._dict.keys(), key=rev_len_sort):
            v = self._dict[k]
            if v % self.base == 0:
                continue
            p = pauliopt.pi * v / 4
            circ >>= pauliopt.Z(p) @ k
        return circ

    @property
    def n_gadgets(self):
        return len(self._dict)

    @property
    def n_t(self):
        """ Number of phases with odd multiple of pi/4 in a circuit """
        return len(filter(lambda x: x % 2 == 1, self._dict.values()))

    def is_id(self, ignore_cliffords=False):
        new_circ = DiagT(self.n_qubits, _dict=self._dict.copy())
        done = False
        while not done:
            done = True
            for k in sorted(new_circ._dict.keys(), key=rev_len_sort):
                v = new_circ._dict[k]
                if len(k) >= 4:
                    new_circ.spider_nest(k, v)
                    done = False
                    break

        if ignore_cliffords or self.ignore_cliffords:
            return all(v % 2 == 0 for v in new_circ._dict.values())
        return new_circ.n_gadgets == 0

    def cloned(self):
        return DiagT(self.n_qubits, _dict=self._dict.copy())
