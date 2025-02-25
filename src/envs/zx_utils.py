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

    def set_type(self, vertex: VT, ty: VertexType) -> None:
        super().set_type(vertex, ty)
        self.verts_changed.add(vertex)

    def set_edge_type(self, e, t):
        super().set_edge_type(e, t)
        self.verts_changed.update(e)

    def add_vertex(self,
                   ty:VertexType=VertexType.BOUNDARY,
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

    def add_edges(self, edges, edgetype=EdgeType.SIMPLE):
        super().add_edges(edges, edgetype)
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

