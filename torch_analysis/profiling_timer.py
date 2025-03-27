import time
import sys
import torch
import statistics
import torch.autograd.profiler as torch_profiler
sys.setrecursionlimit(1500)
from torch.cuda import Event


class Timer:
    """
    A profiling timer for measuring execution time of PyTorch operations.
    """
    def __init__(
        self, profiling_steps: int, name: str, use_ncu=False
    ):
        """
        Initializes the Timer with profiling parameters.
        """
        self.use_ncu = use_ncu
        self.warming = 50
        self.steps = 5
        self.profiling_steps = profiling_steps
        self.name = name
        self.database = dict()
        self.variance = dict()
        self.grad_fn_list = []
        self.grad_fn_input_list = []
        self.backward_op_dict = dict()

    def _init_database(
        self
    ):
        """
        Resets the profiling database.
        """
        self.database = dict()
        self.variance = dict()

    def v1_bp_profiling(
        self
    ):
        """
        Performs backward pass profiling using time.perf_counter.
        """
        for var_name, outputs in zip(self.grad_fn_list, self.grad_fn_input_list):
            name = var_name['name']
            var = var_name['var']
            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                for i in range(self.warming):
                    var(*outputs)
    
                data_list = []
                for _ in range(self.steps):
                    torch.cuda.synchronize()
                    ss = time.perf_counter()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    torch.cuda.synchronize()
                    ee = time.perf_counter()
                    data_list.append((ee-ss))
                self.database[name] = statistics.mean(data_list) / self.profiling_steps
                self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps

    def _bp_profiling(
        self
    ):
        """
        Performs backward pass profiling using CUDA events.
        """
        for var_name, outputs in zip(self.grad_fn_list, self.grad_fn_input_list):
            name = var_name['name']
            var = var_name['var']
            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                for i in range(self.warming):
                    var(*outputs)
                torch.cuda.synchronize()

                data_list = []
                for _ in range(self.steps):
                    start_event = Event(enable_timing=True)
                    end_event = Event(enable_timing=True)

                    start_event.record()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    data_list.append(start_event.elapsed_time(end_event))
                self.database[name] = statistics.mean(data_list) / self.profiling_steps
                self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps

    def _get_bp_node_op(
        self, var
    ) -> str:
        """
        Retrieves the operation name for a backward pass node.
        """
        return type(var).__name__

    def v1_make_hook(
        self, var
    ) -> callable:
        """
        Creates a hook for backward pass profiling using time.perf_counter.
        """
        def hook(inputs, outputs):
            if self._get_bp_node_op(var) not in self.backward_op_dict:
                self.backward_op_dict[self._get_bp_node_op(var)] = 0
            else:
                self.backward_op_dict[self._get_bp_node_op(var)] += 1

            name = self._get_bp_node_op(var) + str(self.backward_op_dict[self._get_bp_node_op(var)])

            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                for i in range(self.warming):
                    var(*outputs)
                
                data_list = []
                for _ in range(self.steps):
                    torch.cuda.synchronize()
                    ss = time.perf_counter()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    torch.cuda.synchronize()
                    ee = time.perf_counter()
                    data_list.append((ee-ss))
                self.database[name] = statistics.mean(data_list) / self.profiling_steps
                self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps
        return hook

    def _make_hook(
        self, var
    ) -> callable:
        """
        Creates a hook for backward pass profiling using CUDA events.
        """
        def hook(
            inputs, outputs
        ) -> None:
            if self._get_bp_node_op(var) not in self.backward_op_dict:
                self.backward_op_dict[self._get_bp_node_op(var)] = 0
            else:
                self.backward_op_dict[self._get_bp_node_op(var)] += 1

            name = self._get_bp_node_op(var) + str(self.backward_op_dict[self._get_bp_node_op(var)])

            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                for i in range(self.warming):
                    var(*outputs)
                torch.cuda.synchronize()

                data_list = []
                for _ in range(self.steps):
                    start_event = Event(enable_timing=True)
                    end_event = Event(enable_timing=True)

                    start_event.record()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    data_list.append(start_event.elapsed_time(end_event))
                
                if self.use_ncu:    
                    print('var', var)
                    torch.cuda.nvtx.range_push("Backward")
                    var(*outputs)
                    torch.cuda.nvtx.range_pop()
                    print(f"finished ncu profiling for {name} during backward pass")

                self.database[name] = statistics.mean(data_list) / self.profiling_steps
                self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps
        return hook

    def _empty_hook(
        self, var
    ) -> callable:
        """
        Creates an empty hook for backward pass.
        """
        def hook(
            inputs, outputs
        ) -> None:
            pass
        return hook

    def v1_call_function(
        self, function, node, args, kwargs
    ):
        """
        :param function: Interpreter.call_module
        :param node: node in symbolic_traced_module.graph.nodes
        :param args: input tensor
        """
        for i in range(self.warming):
            function(node.target, args, kwargs)
        data_list = []
        torch.cuda.synchronize()

        for _ in range(self.steps):
            ss = time.perf_counter()

            for i in range(self.profiling_steps):
                function(node.target, args, kwargs)
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            ee = time.perf_counter()

            data_list.append((ee-ss))
        self.database[node.name] = statistics.mean(data_list) / self.profiling_steps
        self.variance[node.name] = statistics.variance(data_list) / self.steps / self.profiling_steps

        return function(node.target, args, kwargs)


    def _call_function(
        self, function, node, args, kwargs
    ):
        """
        :param function: Interpreter.call_module
        :param node: node in symbolic_traced_module.graph.nodes
        :param args: input tensor
        """
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        for i in range(self.warming):
            function(node.target, args, kwargs)
        data_list = []
        torch.cuda.synchronize()

        for _ in range(self.steps):
            start_event.record()
            for i in range(self.profiling_steps):
                function(node.target, args, kwargs)
            end_event.record()
            torch.cuda.synchronize()

            data_list.append(start_event.elapsed_time(end_event))

        if self.use_ncu:
            print('test'*10)
            torch.cuda.nvtx.range_push("Forward")
            function(node.target, args, kwargs)
            torch.cuda.nvtx.range_pop()
            print(f"finished ncu profiling for {node.name} during forward pass")

        self.database[node.name] = statistics.mean(data_list) / self.profiling_steps
        self.variance[node.name] = statistics.variance(data_list) / self.steps / self.profiling_steps

        return function(node.target, args, kwargs)


    def _call_function_profile(
        self, function, args
    ):
        """
        Profiles a function using PyTorch's autograd profiler.
        """
        function(args)
        with torch_profiler.profile(use_cuda=True) as prof:
            function(args)
        count = 0
        result = 0
        for e in prof.function_events:
            if e.self_cuda_time_total != 0:
                self.database[self.id_list[count]] += e.self_cuda_time_total  / 1e6
                result += e.self_cuda_time_total
                count += 1

    def v1_call_function_once(
        self, function, node, args, kwargs
    ):
        """
        Performs a single forward pass function call and measures execution time using time.perf_counter.
        """
        torch.cuda.synchronize()
        ss = time.perf_counter()
        output = function(node.target, args, kwargs)
        torch.cuda.synchronize()
        ee = time.perf_counter()

        self.database[node.name] = 0
        self.variance[node.name] = 0
        return output
    
    def _call_function_once(
        self, function, node, args, kwargs
    ):
        """
        Performs a single forward pass function call and measures execution time using CUDA events.
        """
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        output = function(node.target, args, kwargs)
        end_event.record()
        torch.cuda.synchronize()

        self.database[node.name] = start_event.elapsed_time(end_event) / 1 / 1
        self.variance[node.name] = start_event.elapsed_time(end_event) / 1
        return output

    def v1_call_optimizer(
        self, function, name
    ):
        """
        Performs optimizer step profiling using time.perf_counter.
        """
        for i in range(self.warming):
            function()
        data_list = []
        for _ in range(self.steps):
            torch.cuda.synchronize()
            ss = time.perf_counter()
            for i in range(self.profiling_steps):
                function()
            torch.cuda.synchronize()
            ee = time.perf_counter()
            data_list.append((ee-ss))
        self.database[name] = statistics.mean(data_list) / self.profiling_steps
        self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps


    def _call_optimizer(
        self, function, name
    ):
        """
        Performs optimizer step profiling using CUDA events.
        """
        for i in range(self.warming):
            function()
        data_list = []
        torch.cuda.synchronize()

        for _ in range(self.steps):
            start_event = Event(enable_timing=True)
            end_event = Event(enable_timing=True)
            start_event.record()
            for i in range(self.profiling_steps):
                function()
                torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            data_list.append(start_event.elapsed_time(end_event))

        self.database[name] = statistics.mean(data_list) / self.profiling_steps
        self.variance[name] = statistics.variance(data_list) / self.steps / self.profiling_steps

    def _get_database(
        self
    ):
        return self.database

    def _get_variance(
        self
    ):
        return self.variance


def make_dot(
    var, params, hook
):
    """
    Produces Graphviz representation of PyTorch autograd graph.
    """
    param_map = {id(v): k for k, v in params}

    seen = set()
    
    def add_nodes(
        var
    ):
        if var not in seen:
            node_id = str(id(var))
            var.register_hook(hook(var))
            seen.add(var)

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t)
    add_nodes(var.grad_fn)