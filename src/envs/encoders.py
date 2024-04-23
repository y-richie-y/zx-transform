from abc import ABC, abstractmethod
from collections import OrderedDict
from random import random
import pyzx
from .diag_t import DiagT

from src.envs.zx_utils import TrackingGraph, flat_reduce

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    def decode(self, lst):
        v, p = self.parse(lst)
        if p == 0:
            return None
        return v


class Graph(Encoder):
    """
    graph encoder
    """
    def __init__(self, params):
        super().__init__()
        self.log_qubits_depth = params.log_qubits_depth
        self.precise = params.precise_vocab
        self.symbols = (
            [str(i) for i in range(params.max_int + 1)] +
            [f"N{i}" for i in range(params.max_nodes + 1)]
        )
        if self.precise:
            self.symbols.extend(["E1", "E2", "T0", "T1", "T2", "T3"])

    def encode(self, circ, qubits=None, depth=None):
        """ Converts pyzx graph into tokens. """
        indices = sorted(circ.types().keys())
        types_dict = OrderedDict(circ.types())
        phases_dict = OrderedDict(circ.phases())
        nr_nodes = len( circ.types())
        assert nr_nodes == circ.num_vertices()
        nr_edges = circ.num_edges()
        if self.log_qubits_depth:
            step = circ.current_step
            toks = [str(qubits), str(depth), str(step), str(nr_nodes), str(nr_edges)]
        else:
            toks = [str(nr_nodes), str(nr_edges)]
            
        for key in types_dict.keys():
            if self.precise:
                toks.extend([f"T{types_dict[key]}", str(int(phases_dict[key] * 4))]) 
            else:
                toks.extend([str(types_dict[key]), str(int(phases_dict[key] * 4))]) 
        for edge in circ.edges():
            if self.precise:
                toks.append(f"E{circ.edge_type(edge)}")
            else:
                toks.append(str(circ.edge_type(edge)))
            toks.extend([f"N{indices.index(x)}" for x in edge])
        return toks

    def parse(self, lst):
        offset = 3 if self.log_qubits_depth else 0
        if len(lst) < 2 + offset:
            return None, 0
        graph = pyzx.Graph()
        input_time = True
        inputs = []
        outputs = []
        try: 
            nr_nodes = int(lst[offset])
            nr_edges = int(lst[offset + 1])
            if len(lst) != 2 + offset + 2 * nr_nodes + 3 * nr_edges:
                return None, 0
            offset += 2
            for _ in range(nr_nodes):
                if self.precise:
                    type = int(lst[offset][1:])
                else:
                    type = int(lst[offset])
                phase = int(lst[offset+1])
                v = graph.add_vertex(phase=phase/4, ty=type )
                if type == 0:
                    if input_time: 
                        graph.set_position(v, q=len(inputs), r=0)
                        inputs.append(v)
                    else: 
                        graph.set_position(v, q=len(outputs), r=len(inputs))
                        outputs.append(v)
                else:
                    input_time = False
                    x = 0.5 + random() * (len(inputs) - 1)
                    y = 0.5 + random() * (len(inputs) - 1)
                    graph.set_position(v, q=x, r=y)
                offset += 2 
            graph.set_inputs(inputs)
            graph.set_outputs(outputs)
            for _ in range(nr_edges):
                if self.precise:
                    type = int(lst[offset][1:])
                else:
                    type = int(lst[offset])
                edge = [int(s[1:]) for s in lst[offset+1:offset+3]]
                graph.add_edge(edge, type)
                offset += 3
        except Exception as e:
            #print(e)
            return None, 0
        graph = TrackingGraph.upgrade(graph)
        return  graph, offset


class NewCircuit(Encoder):
    """ Binary encoding the legs of the gadget """
    def __init__(self, params):
        super().__init__()
        self.log_depth = params.log_depth
        self.precise = True
        self.symbols = [f"nqubits={i}" for i in range(params.max_int + 1)]
        # whether the gadget acts on that leg
        self.symbols.extend(["YES", "NO"])
        self.symbols.extend([f"Z({i}T)" for i in range(8)])

    def encode(self, circ, qubits=None, depth=None):
        """ Convert DiagT circuit into tokens. """
        tokens = []
        for qs, phase8 in sorted(circ._dict.items()):
            for i in range(circ.n_qubits):
                tokens.append("YES" if i in qs else "NO")
            tokens.append(f"Z({phase8}T)")
        return tokens

    def parse(self, tokens):
        n_qubits = int(tokens[0].split("=")[1])
        circ = DiagT(n_qubits)
        legs = set()
        i = 0
        for token in tokens[1:]:
            if token in ["YES", "NO"]:
                if token == "YES":
                    legs.add(i)
                i += 1
            else:
                # basis = token[0]
                angle8 = int(token[2:-2])
                circ.add_gadget(angle8, legs)
                legs = set()
                i = 0

        # return a non zero value to indicate success
        offset = -1
        return circ, offset


class Circuit(Encoder):
    def __init__(self, params):
        super().__init__()
        self.log_depth = params.log_depth
        # TODO: what is precise?
        self.precise = True
        self.symbols = [f"nqubits={i}" for i in range(params.max_int + 1)]
        self.symbols.extend([f"q{i}" for i in range(params.max_int + 1)])
        self.symbols.extend([f"Z({i}T)" for i in range(8)])
        # TODO not ignore cliffords setting

    def encode(self, circ, qubits=None, depth=None):
        """ Convert DiagT circuit into tokens. """
        tokens = [f"nqubits={circ.n_qubits}"]
        for qs, phase8 in sorted(circ._dict.items()):
            tokens.extend([f"q{i}" for i in qs])
            tokens.append(f"Z({phase8}T)")
        return tokens

    def parse(self, tokens):
        n_qubits = int(tokens[0].split("=")[1])
        circ = DiagT(n_qubits)
        legs = set()
        for token in tokens[1:]:
            if token.startswith("q"):
                legs.add(int(token[1:]))
            else:
                # basis = token[0]
                angle8 = int(token[2:-2])
                circ.add_gadget(angle8, legs)
                legs = set()

        # return a non zero value to indicate success
        offset = -1
        return circ, offset
