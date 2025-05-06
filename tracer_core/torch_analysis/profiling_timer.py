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
        self, profiling_steps: int, name: str, use_ncu=False, logger=None
    ):
        """
        Initializes the Timer with profiling parameters.
        
        Args:
            profiling_steps: Number of times to repeat each operation for more accurate timing
            name: Name of the model or module being profiled
            use_ncu: Whether to use NVIDIA Compute Unified Architecture profiling
            logger: Logger for outputting profiling information
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
        self.logger = logger

    def _init_database(
        self
    ):
        """
        Resets the profiling database.
        """
        self.database = dict()
        self.variance = dict()

    def _bp_profiling(
        self
    ):
        """
        Performs backward pass profiling using CUDA events.
        """
        if self.logger:
            self.logger.info("Starting backward pass profiling")
            
        for idx, (var_name, outputs) in enumerate(zip(self.grad_fn_list, self.grad_fn_input_list)):
            name = var_name['name']
            var = var_name['var']
            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                for i in range(self.warming):
                    var(*outputs)
                torch.cuda.synchronize()

                data_list = []
                for run in range(self.steps):    
                    start_event = Event(enable_timing=True)
                    end_event = Event(enable_timing=True)

                    start_event.record()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    elapsed_time = start_event.elapsed_time(end_event)
                    data_list.append(elapsed_time)
                    
                mean_time = statistics.mean(data_list) / self.profiling_steps
                variance = statistics.variance(data_list) / self.steps / self.profiling_steps
                
                self.database[name] = mean_time
                self.variance[name] = variance
                
                if self.logger:
                    self.logger.info(f"operation: {name:<25} running time: {mean_time:<5.6f} ± {variance**0.5:<10.6f} ms")

    def _get_bp_node_op(
        self, var
    ) -> str:
        """
        Retrieves the operation name for a backward pass node.
        """
        return type(var).__name__


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
                # Warming phase
                if self.logger:
                    self.logger.debug(f"Warming up {self.warming} iterations for {name}")
                for i in range(self.warming):
                    var(*outputs)
                torch.cuda.synchronize()

                # Actual profiling phase
                if self.logger:
                    self.logger.debug(f"Starting {self.steps} profiling runs for {name}, each with {self.profiling_steps} iterations")
                
                data_list = []
                for run in range(self.steps):
                    start_event = Event(enable_timing=True)
                    end_event = Event(enable_timing=True)

                    start_event.record()
                    for i in range(self.profiling_steps):
                        var(*outputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    elapsed_time = start_event.elapsed_time(end_event)
                    data_list.append(elapsed_time)
                    
                if self.use_ncu:    
                    if self.logger:
                        self.logger.info(f"Running NCU profiling for {name}")
                    torch.cuda.nvtx.range_push("Backward")
                    var(*outputs)
                    torch.cuda.nvtx.range_pop()
                    if self.logger:
                        self.logger.info(f"Finished NCU profiling for {name} during backward pass")

                # Calculate statistics
                mean_time = statistics.mean(data_list) / self.profiling_steps
                variance = statistics.variance(data_list) / self.steps / self.profiling_steps
                
                self.database[name] = mean_time
                self.variance[name] = variance
                
                if self.logger:
                    self.logger.info(f"operation: {name:<25} running time: {mean_time:<5.6f} ± {variance**0.5:<10.6f} ms")
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

    def _call_function(
        self, function, node, args, kwargs
    ):
        """
        Profiles the execution time of a forward pass function.
        
        Args:
            function: Function to be profiled (usually from Interpreter)
            node: Node in symbolic_traced_module.graph.nodes
            args: Input tensors
            kwargs: Keyword arguments
            
        Returns:
            The result of calling the function with the provided arguments
        """
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        # Warming phase
        if self.logger:
            self.logger.debug(f"Warming up {self.warming} iterations for {node.name}")
        for i in range(self.warming):
            function(node.target, args, kwargs)
        data_list = []
        torch.cuda.synchronize()

        # Actual profiling phase
        if self.logger:
            self.logger.debug(f"Starting {self.steps} profiling runs for {node.name}, each with {self.profiling_steps} iterations")
        
        for run in range(self.steps):

            start_event.record()
            for i in range(self.profiling_steps):
                function(node.target, args, kwargs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            data_list.append(elapsed_time)
            

        if self.use_ncu:
            if self.logger:
                self.logger.info(f"Running NCU profiling for {node.name}")
            torch.cuda.nvtx.range_push("Forward")
            function(node.target, args, kwargs)
            torch.cuda.nvtx.range_pop()
            if self.logger:
                self.logger.info(f"Finished NCU profiling for {node.name} during forward pass")

        # Calculate statistics
        mean_time = statistics.mean(data_list) / self.profiling_steps
        variance = statistics.variance(data_list) / self.steps / self.profiling_steps
        
        self.database[node.name] = mean_time
        self.variance[node.name] = variance
        
        if self.logger:
            self.logger.info(f"operation: {node.name:<25} running time: {mean_time:<5.6f} ± {variance**0.5:<10.6f} ms")

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

    def _call_function_once(
        self, function, node, args, kwargs
    ):
        """
        Performs a single forward pass function call and measures execution time using CUDA events.
        """
        if self.logger:
            self.logger.debug(f"Single profiling run for operation: {node.name}")
            
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        output = function(node.target, args, kwargs)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        self.database[node.name] = elapsed_time
        self.variance[node.name] = elapsed_time
        
        if self.logger:
            self.logger.info(f"Operation {node.name:<30} running time: {elapsed_time:<15.6f} ms")

        return output

    def _call_optimizer(
        self, function, name
    ):
        """
        Performs optimizer step profiling using CUDA events.
        """
        # if self.logger:
        #     self.logger.info(f"Profiling optimizer: {name}")
            
        for i in range(self.warming):
            function()
        data_list = []
        torch.cuda.synchronize()

        if self.logger:
            self.logger.debug(f"Starting {self.steps} profiling runs for {name}, each with {self.profiling_steps} iterations")
            
        for run in range(self.steps):
            if self.logger and run == 0:
                self.logger.debug(f"Run {run+1}/{self.steps} for {name}")
                
            start_event = Event(enable_timing=True)
            end_event = Event(enable_timing=True)
            start_event.record()
            for i in range(self.profiling_steps):
                function()
                torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            data_list.append(elapsed_time)
            
            if self.logger and run == self.steps - 1:
                self.logger.debug(f"running time run {run+1}/{self.steps} for {name}")

        mean_time = statistics.mean(data_list) / self.profiling_steps
        variance = statistics.variance(data_list) / self.steps / self.profiling_steps
        
        self.database[name] = mean_time
        self.variance[name] = variance
        
        if self.logger:
            self.logger.info(f"Optimizer {name:<25} running time: {mean_time:<3.4f} ± {variance**0.5:<10.4f} ms")

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