# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators
from src.dataset import EnvDataset


from torch.utils.data import DataLoader
from pyzx.graph.graph_s import GraphS
from pyzx.graph.base import VT, ET
from pyzx.utils import VertexType, EdgeType, FractionLike, FloatInt
from pyzx.simplify import spider_simp, full_reduce, to_gh
from pyzx.simplify import BaseGraph, Stats


from typing import List, Mapping, Optional, Tuple

from ..utils import bool_flag


SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

class PyzxEnvironment(object):

    TRAINING_TASKS = {"simplify"}

    def __init__(self, params):
        self.max_len = params.max_len
        
        self.encoder = encoders.Graph(params)
        self.generator = generators.Graphs(params)
        self.generator = generators.FlatGraphs(params)
        self.log_qubits_depth = params.log_qubits_depth

        # vocabulary
        self.words = SPECIAL_WORDS + sorted(list(set(self.encoder.symbols)))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

    # TODO
    def input_to_infix(self, lst):
        return ''.join(lst)
    # TODO
    def output_to_infix(self, lst):
        return ''.join(lst)
        
    def gen_expr(self, data_type=None, task=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        gen = self.generator.generate(self.rng)
        if gen is None:
            return None
        x_data, y_data, qubits, depth = gen
        # encode input
        x = self.encoder.encode(x_data, qubits, depth)
        # encode output
        y = self.encoder.encode(y_data, qubits, depth)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        if self.log_qubits_depth:
            d = i % 100
            q = (i // 100) % 100
            s = i // 10000
            return f"{s=}/{q=}/{d=}"
        return f"{i*100}-{(i+1)*100}"

    def code_class(self, xi, yi):
        if self.log_qubits_depth:
            return int(xi[2]) * 10000 + int(xi[0]) * 100 + int(xi[1])
        return int(len(xi))//100

    def check_prediction(self, src, tgt, hyp):
        h = self.encoder.decode(hyp)
        s = self.encoder.decode(src)
        t = self.encoder.decode(tgt)
        if h is None:
            return -1,-1,-1,-1
        return self.generator.evaluate(s,t,h)

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--min_qubits", type=int, default=1, help="min nr of cubits"
        )
        parser.add_argument(
            "--max_qubits", type=int, default=20, help="maxnr of cubits"
        )
        parser.add_argument(
            "--qubit_step", type=int, default=1, help="qubit step"
        )
        parser.add_argument(
            "--min_depth", type=int, default=1, help="minimum depth"
        )
        parser.add_argument(
            "--max_depth", type=int, default=20, help="maximum depth"
        )
        parser.add_argument(
            "--depth_step", type=int, default=1, help="depth step"
        )
        
        parser.add_argument(
            "--log_qubits_depth", type=bool_flag, default=False, help="log nr of qbits and depths in the 2 first tokens"
        )
        parser.add_argument(
            "--precise_vocab", type=bool_flag, default=False, help="precise encoding vocabulary"
        )
        
        
        parser.add_argument(
            "--max_int", type=int, default=1000, help="maximum depth"
        )
        parser.add_argument(
            "--max_nodes", type=int, default=1000, help="maximum depth"
        )
        parser.add_argument(
            "--circuit_type", type=str, default="clifford", help="type of circuit to generate, clifford, cnot, cliffordT"
        )


class PaulioptEnvironment(object):
    TRAINING_TASKS = {"simplify"}

    def __init__(self, params):
        self.max_len = params.max_len

        self.encoder = encoders.Circuit(params)
        self.generator = generators.Circuits(params)
        self.log_depth = params.log_depth

        # vocabulary
        self.words = SPECIAL_WORDS + sorted(list(set(self.encoder.symbols)))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

    # TODO
    def input_to_infix(self, lst):
        return ''.join(lst)

    # TODO
    def output_to_infix(self, lst):
        return ''.join(lst)

    def gen_expr(self, data_type=None, task=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        gen = self.generator.generate(self.rng)
        if gen is None:
            return None
        x_data, y_data, qubits, depth = gen
        # encode input
        x = self.encoder.encode(x_data, qubits, depth)
        # encode output
        y = self.encoder.encode(y_data, qubits, depth)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        if self.log_depth:
            return f"{i//100}/{i%100}"
        return f"{i*100}-{(i+1)*100}"

    def code_class(self, xi, yi):
        if self.log_depth:
            return int(xi[0]) * 100 + int(xi[1])
        return int(len(xi))//100

    def check_prediction(self, src, tgt, hyp):
        h = self.encoder.decode(hyp)
        s = self.encoder.decode(src)
        t = self.encoder.decode(tgt)
        if h is None:
            return -1, -1, -1, -1
        return self.generator.evaluate(s, t, h)

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--min_qubits", type=int, default=1, help="min nr of cubits"
        )
        parser.add_argument(
            "--max_qubits", type=int, default=20, help="maxnr of cubits"
        )
        parser.add_argument(
            "--qubit_step", type=int, default=1, help="qubit step"
        )
        parser.add_argument(
            "--min_depth", type=int, default=1, help="minimum depth"
        )
        parser.add_argument(
            "--max_depth", type=int, default=20, help="maximum depth"
        )
        parser.add_argument(
            "--depth_step", type=int, default=1, help="depth step"
        )
        parser.add_argument(
            "--walk_steps", type=int, default=1, help="Length of random walk"
        )

        parser.add_argument(
            "--log_depth", type=bool_flag, default=False, help="log nr of qbits and depths in the 2 first tokens"
        )

        parser.add_argument(
            "--max_int", type=int, default=1000, help="maximum depth"
        )


