import json
from typing import Tuple
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
from timer import Timer, make_dot
from typing import Iterator, Any
from typing import Dict, List
import torch.optim as optim
import torch.autograd.profiler as torch_profiler
from torch.fx.experimental.proxy_tensor import FakeTensor, make_fx
import os


class TorchDatabase(torch.fx.Interpreter):
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
    def __init__(self, module: torch.nn.Module, example: torch.tensor, name: str, timer: Timer, optimizer: Optimizer):
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

        def register_last_uses(n: Node, user: Node):
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
        print(f"Start fwd profiling...")
        self._get_fp_node_time()
        torch.cuda.synchronize()

        del self.env
        print(f"Start bwd profiling...")
        self._get_bp_node_time()
        torch.cuda.synchronize()
        
        print(f"Start optimizer profiling...")
        self._get_optimizer_node_time()
        torch.cuda.synchronize()

    
    def _fp_node_run(self, node: torch.fx.node.Node, *args):
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
                # placeholder optime is fixed, thus only need once profiling
                self.env[node] = self.timer._call_function_once(attr, node, args, kwargs)
            else:
                self.env[node] = self.timer._call_function(attr, node, args, kwargs)

            for to_delete in self.user_to_last_uses.get(node, []):
                del self.env[to_delete]

        self._forward_database = self.timer._get_database()
        self._forward_variance = self.timer._get_variance()
        
    def _get_fp_node_time(self, initial_env = None):
        self.attr = {}
        for node in self._symbolic_traced_module.graph.nodes:
            if node in self.env:
                continue

            if node.name in self._forward_database:
                continue
            else:
                self._fp_node_run(node, self.example)

    def _get_bp_node_time(self):
        self.timer._init_database()
        # if isinstance(self.module, PreTrainedModel):
        #     y = self.module(self.example)
        #     if 'pooler_output' in y.__dict__:
        #         y = y.pooler_output
        #     else:
        #         y = y.last_hidden_state
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

    # def _get_all_node_time(self):
    #     # self.timer._init_database()
    #     self.timer._all_profiling(self.module,self.example)
    #     self._forward_database = self.timer._get_database()
    #     self._forward_variance = self.timer._get_variance()
    #     self._backward_database = {}
    #     self._backward_variance = {}

    def _get_optimizer_node_time(self):
        self.timer._init_database()
        # self.timer._call_optimizer(self.optimizer.zero_grad, "optimizer_zero")
        self.timer._call_optimizer(self.optimizer.step, "optimizer_step")
        self._optimizer_database = self.timer.database
        self._optimizer_variance = self.timer._get_variance()

    def _get_overall_database(self):
        self._overall_database = {**self._forward_database, **self._backward_database, **self._optimizer_database}
        return self._overall_database

    def _get_overall_variance(self):
        self._overall_variance = {**self._forward_variance, **self._backward_variance, **self._optimizer_variance}
        return self._overall_variance

    def _get_fwd_database(self):
        return self._forward_database

    def _get_bwd_database(self):
        return self._backward_database

    def _get_optimizer_database(self):
        return self._optimizer_database

    def dump_runtime(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_overall_database(), file, indent=4)

    def dump_fwd_runtime(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_fwd_database(), file, indent=4)

    def dump_bwd_runtime(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_bwd_database(), file, indent=4)

    def dump_optim_runtime(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self._get_optimizer_database(), file, indent=4)