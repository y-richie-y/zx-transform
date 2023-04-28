from abc import ABC, abstractmethod
import numpy as np

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
        self.prefix = prefix
        self.symbols = [self.prefix + str(i) for i in range(min, max+1)]

    def encode(self, value):
        return [self.prefix+str(value)]

    def parse(self, lst):
        if len(lst) == 0 or (not lst[0] in self.symbols):
            return None, 0
        return  int(lst[0][len(self.prefix):]), 1


