import json
from typing import List, Optional, Dict, Any, Iterator, Tuple
import torch
import torch.fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx import symbolic_trace, Interpreter
from torch.optim.optimizer import Optimizer
from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
from shape_prop import ShapeProp, TensorMetadata
from typename import typename
from Node import Node
from transformers import PreTrainedModel
import time
from profiling_timer import Timer, make_dot
import torch.optim as optim
import torch.autograd.profiler as torch_profiler
from torch.fx.experimental.proxy_tensor import FakeTensor, make_fx
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TorchDatabase(
    torch.fx.Interpreter
):
    """
    Generate a torch database with torch.fx.Interpreter
    Basic usage:
        module = torchvision.models.resnet50(pretrained=True).cuda()
        example = torch.rand(1, 3, 224, 224).cuda()
        optimizer = optim.SGD(module.parameters(), lr=0.01)

        timer = Timer(100, 'resnet50')

        g = TorchDatabase(module, example, 'resnet50', 100)
        print(g._forward_database) 
        print(g._backward_database) 
        print(g._optimizer_database) 
    """
    def __init__(
        self, module: torch.nn.Module, example: torch.tensor, name: str, timer: Timer, optimizer: Optimizer
    ) -> None:
        self.module = module
        self.example = example
        self.name = name
        self.timer = timer
        self.optimizer = optimizer
        if isinstance(self.module, PreTrainedModel):
            self._symbolic_traced_module = transformers_symbolic_trace(self.module)
        else:
            self._symbolic_traced_module = symbolic_trace(self.module)

        self.submodules = dict(self._symbolic_traced_module.named_modules())

        node_to_last_use: Dict[Node, Node] = {}
        self.user_to_last_uses: Dict[Node, List[Node]] = {}

        def register_last_uses(
            n: Node, user: Node
        ) -> None:
            """
            Register the last use of a node
            """
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                self.user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self._symbolic_traced_module.graph.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        self._forward_database = {}
        self._backward_database = {}
        self._optimizer_database = {}
        self._overall_database = {}
    
        self._forward_variance = {}
        self._backward_variance = {}
        self._optimizer_variance = {}
        self._overall_variance = {} 

        initial_env = None
        self.env = initial_env if initial_env else {}
        logging.info("Start fwd profiling...")
        print(f"{'Operation':<30} {'Runtime (ms)':<25}")
        self._get_fp_node_time()
        torch.cuda.synchronize()
        logging.info("torch_graph: fwd profiling completed...")
        print("-" * 90)

        del self.env
        logging.info("Start bwd profiling...")
        print(f"{'Operation':<30} {'Runtime (ms)':<25}")
        self._get_bp_node_time()
        torch.cuda.synchronize()
        logging.info("torch_graph: bwd profiling completed...")
        print("-" * 90)
        
        logging.info("Start optimizer profiling...")
        print(f"{'Operation':<30} {'Runtime (ms)':<25}")
        self._get_optimizer_node_time()
        torch.cuda.synchronize()
        logging.info("torch_graph: fbwd profiling completed...")
        print("-" * 90)

    
    def _fp_node_run(
        self, node: torch.fx.node.Node, *args
    ) -> None:
        """
        Run a single node in the forward pass
        """
        self.args_iter: Iterator[Any] = iter(args)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        if node.op in self.attr:
            attr = self.attr[node.op]
        else:
            attr = getattr(self, node.op)
            self.attr[node.op] = attr

        if node.name in self._forward_database:
            raise RuntimeError(f"Node {node} repeat in {self.name} graph")
        else:
            if node.op == "placeholder":
                self.env[node] = self.timer._call_function_once(attr, node, args, kwargs)
            else:
                self.env[node] = self.timer._call_function(attr, node, args, kwargs)

            for to_delete in self.user_to_last_uses.get(node, []):
                del self.env[to_delete]

        self._forward_database = self.timer._get_database()
        self._forward_variance = self.timer._get_variance()
        
    def _get_fp_node_time(
        self, initial_env = None
    ) -> None:
        """
        Get the runtime of each node in the forward pass
        """
        self.attr = {}
        for node in self._symbolic_traced_module.graph.nodes:
            if node in self.env:
                continue

            if node.name in self._forward_database:
                continue
            else:
                self._fp_node_run(node, self.example)
                
        for node_name, runtime in self._forward_database.items():
            print(f"{node_name:<30} {runtime:<25.20f}")

    def _get_bp_node_time(
        self
    ) -> None:
        """
        Get the runtime of each node in the backward pass
        """
        self.timer._init_database()

        if isinstance(self.module, PreTrainedModel):
            y = self.module(self.example)
            if isinstance(y, tuple):
                y = y[0]  # adjust this as needed
            if hasattr(y, 'pooler_output'):
                y = y.pooler_output
            elif hasattr(y, 'last_hidden_state'):
                y = y.last_hidden_state
        else:
            y = self.module(self.example)
        
        make_dot(y, self.module.named_parameters(), self.timer._make_hook)
        y.backward(y)
        self._backward_database = self.timer._get_database()
        self._backward_variance = self.timer._get_variance()

        # Print the backward pass runtime
        for node_name, runtime in self._backward_database.items():
            print(f"{node_name:<30} {runtime:<25.20f}")

    def _get_optimizer_node_time(
        self
    ) -> None:
        """
        Get the runtime of each node in the optimizer
        """
        self.timer._init_database()
        # self.timer._call_optimizer(self.optimizer.zero_grad, "optimizer_zero")
        self.timer._call_optimizer(self.optimizer.step, "optimizer_step")
        self._optimizer_database = self.timer.database
        self._optimizer_variance = self.timer._get_variance()

        # Print the optimizer step runtime
        for node_name, runtime in self._optimizer_database.items():
            print(f"{node_name:<30} {runtime:<25.20f}")

    def _get_overall_database(
        self
    ) -> Dict[str, float]:
        """
        Get the overall runtime of each node in the forward pass, backward pass and optimizer
        """
        self._overall_database = {**self._forward_database, **self._backward_database, **self._optimizer_database}
        return self._overall_database

    def _get_overall_variance(
        self
    ) -> Dict[str, float]:
        """
        Get the overall variance of each node in the forward pass, backward pass and optimizer
        """
        self._overall_variance = {**self._forward_variance, **self._backward_variance, **self._optimizer_variance}
        return self._overall_variance

    def _get_fwd_database(
        self
    ) -> Dict[str, float]:
        """
        Get the forward pass runtime of each node
        """
        return self._forward_database

    def _get_bwd_database(
        self
    ) -> Dict[str, float]:
        """
        Get the backward pass runtime of each node
        """
        return self._backward_database

    def _get_optimizer_database(
        self
    ) -> Dict[str, float]:
        """
        Get the optimizer runtime of each node
        """
        return self._optimizer_database

    def dump_runtime(
        self, path
    ) -> None:
        """
        Dump the overall runtime of each node in the forward pass, backward pass and optimizer
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_overall_database(), file, indent=4)

    def dump_fwd_runtime(
        self, path
    ) -> None:
        """
        Dump the forward pass runtime of each node
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_fwd_database(), file, indent=4)

    def dump_bwd_runtime(
        self, path
    ) -> None:
        """
        Dump the backward pass runtime of each node
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_bwd_database(), file, indent=4)

    def dump_optim_runtime(
        self, path
    ) -> None:
        """
        Dump the optimizer runtime of each node
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_optimizer_database(), file, indent=4)