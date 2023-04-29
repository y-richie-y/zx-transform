from abc import ABC, abstractmethod
import numpy as np
from collections import OrderedDict
import pyzx

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
        self.symbols = [str(i) for i in range(params.max_int + 1)] + [f"N{i}" for i in range(params.max_nodes + 1)]

    def encode(self, circ):
        """ Converts pyzx graph into tokens. """
        indices = sorted(circ.types().keys())
        types_dict = OrderedDict(circ.types())
        phases_dict = OrderedDict(circ.phases())
        nr_nodes = len( circ.types)
        nr_edges = len(circ.edges())
        toks = [str(nr_nodes), str(nr_edges)]
        for typ, phase in zip(types_dict,phases_dict):
            toks.extend([str(typ), str(int(phase * 4))]) 
        for edge in circ.edges():
            toks.append(circ.edge_type(edge))
            toks.extend([f"N{indices.index(x)}" for x in edge])
        return toks

    def parse(self, lst):
        if len(lst) < 2:
            return None, 0
        graph = pyzx.Graph()
        input_time = True
        inputs = []
        outputs = []
        try: 
            nr_nodes = int(lst[0])
            nr_edges = int(lst[1])
            if len(lst) != 2+ 2 * nr_nodes + 3 * nr_edges:
                return None, 0
            offset = 2
            for _ in range(nr_nodes):
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
                    x = 0.5 + np.random.rand() * (len(inputs) - 1)
                    y = 0.5 + np.random.rand() * (len(inputs) - 1)
                    graph.set_position(v, q=x, r=y)
                offset += 2 
            graph.set_inputs(inputs)
            graph.set_outputs(outputs)
            for _ in range(nr_edges):
                type = int(lst[offset])
                edge = list(map(int,lst[offset+1:2]))
                graph.add_edge(edge, type)
                offset += 3
        except Exception as e:
            return None, 0
        return  graph, offset


