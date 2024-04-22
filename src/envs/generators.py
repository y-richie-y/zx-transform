
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
from pyzx.graph.graph_s import GraphS
from pyzx.graph.base import VT, ET
from pyzx.utils import VertexType, EdgeType, FractionLike, FloatInt
from pyzx.simplify import spider_simp, full_reduce, to_gh
from pyzx.simplify import BaseGraph, Stats


from typing import List, Mapping, Optional, Tuple


logger = getLogger()


class TrackingGraph(GraphS):
    """A graph that keeps track of which vertices are added to it."""
    verts_changed = set()

    def reset(self):
        self.verts_changed = set()

    def set_type(self, vertex: VT, ty: VertexType.Type) -> None:
        super().set_type(vertex, ty)
        self.verts_changed.add(vertex)

    def set_edge_type(self, e, t):
        super().set_edge_type(e, t)
        self.verts_changed.update(e)

    def add_vertex(self,
                   ty:VertexType.Type=VertexType.BOUNDARY,
                   qubit:FloatInt=-1,
                   row:FloatInt=-1,
                   phase:Optional[FractionLike]=None,
                   ground:bool=False
                   ) -> VT:
        v = super().add_vertex(ty, qubit, row, phase, ground)
        self.verts_changed.add(v)
        return v

    def add_to_phase(self, vertex: VT, phase: FractionLike) -> None:
        super().add_to_phase(vertex, phase)
        self.verts_changed.add(vertex)

    def add_edges(self, edges, edgetype=EdgeType.SIMPLE, smart=False):
        super().add_edges(edges, edgetype, smart)
        self.verts_changed.update(set([v for e in edges for v in e]))

    def add_vertices(self, amount):
        vs = super().add_vertices(amount)
        new_verts = range(self.num_vertices()-amount, self.num_vertices())
        self.verts_changed.update(new_verts)
        return vs

    def remove_vertices(self, vertices):
        super().remove_vertices(vertices)
        self.verts_changed.update(vertices)

    def remove_vertex(self, vertex):
        super().remove_vertex(vertex)
        self.verts_changed.update(vertex)

    def remove_edges(self, edges):
        super().remove_edges(edges)
        self.verts_changed.update(set([v for e in edges for v in e]))

    def add_edge_table(self, etab: Mapping[Tuple[int, int], List[int]]) -> None:
        for k, _ in etab.items():
            self.verts_changed.update(set(k))
        super().add_edge_table(etab)

    @classmethod
    def upgrade(cls, g: GraphS) -> "TrackingGraph":
        cpy = cls()
        for v, d in g.graph.items():
            cpy.graph[v] = d.copy()
        cpy._vindex = g._vindex
        cpy.nedges = g.nedges
        cpy.ty = g.ty.copy()
        cpy._phase = g._phase.copy()
        cpy._qindex = g._qindex.copy()
        cpy._maxq = g._maxq
        cpy._rindex = g._rindex.copy()
        cpy._maxr = g._maxr
        cpy._vdata = g._vdata.copy()
        cpy.scalar = g.scalar.copy()
        cpy._inputs = tuple(list(g._inputs))
        cpy._outputs = tuple(list(g._outputs))
        cpy.track_phases = g.track_phases
        cpy.phase_index = g.phase_index.copy()
        cpy.phase_master = g.phase_master
        cpy.phase_mult = g.phase_mult.copy()
        cpy.max_phase_index = g.max_phase_index
        return cpy


def flat_reduce(g: BaseGraph[VT,ET], quiet:bool=True, stats:Optional[Stats]=None):
    spider_simp(g, quiet=quiet, stats=stats)
    if g.verts_changed:
        return

    if any(g.type(v) == VertexType.X for v in g.vertices()):
        to_gh(g)
        return

    def matchf(*vs):
        return not any(v in g.verts_changed for v in vs)

    full_reduce(g, matchf=matchf, quiet=quiet, stats=stats)



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


class FlatGraphs(Graphs):
    def __init__(self, params):
        super().__init__(params)
        self.current_graph = None

    def new_graph(self, rng):
        qubits = rng.choice(range(self.min_qubits, self.max_qubits + 1, self.qubit_step))
        depth = rng.choice(range(self.min_depth, self.max_depth + 1, self.depth_step))
        if self.circuit_type == "clifford":
            circ = cliffords(qubits, depth)
        elif self.circuit_type == "cnot":
            circ = cnots(qubits, depth)
        else:
            circ = cliffordT(qubits, depth)
        self.current_graph = TrackingGraph.upgrade(circ)
        self.n_qubits = qubits
        self.n_depth = depth

    def generate(self, rng):
        if self.current_graph is None:
            self.new_graph(rng)
        circ1 = self.current_graph.copy()
        circ2 = self.current_graph

        flat_reduce(circ2)
        if not circ2.verts_changed:
            self.new_graph(rng)
            return self.generate(rng)

        circ2.verts_changed = set()
        return circ1, circ2, self.n_qubits, self.n_depth

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
  
        e = hyp.cloned().merge(tgt.adjoint())

        if e.is_id():
            n_s = src.n_gadgets
            n_t = src.n_gadgets
            n_h = src.n_gadgets
            return 1, n_h, n_t, n_s
        return 0, 0, 0, 0
