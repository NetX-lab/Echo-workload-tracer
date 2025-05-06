#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Tracer - Implementation of the tracer for PyTorch models.
"""
from tracer_core import BaseTracer
from tracer_core.torch_analysis import Timer, TorchGraph, TorchDatabase
from collections import defaultdict
from utils.common import (
    torch, os, json,
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PYTORCH_OPS_PROFILING, PYTORCH_GRAPH_PROFILING, PYTORCH_ONLY_COMPUTE_WORKLOAD,
    Any, Dict, Optional, Union
)

class PyTorchTracer(BaseTracer):
    """
    Tracer implementation for PyTorch models.
    
    This tracer captures runtime and graph information for PyTorch models
    during both forward and backward passes using TorchGraph and TorchDatabase.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: Dict[str, str],
        parallel_setting: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_repeats: int = 10,
        **kwargs
    ) -> None:
        """
        Initialize the PyTorch tracer.
        
        Args:
            model: PyTorch model to trace
            model_name: Name of the model
            example_input: Example input tensor for the model
            output_path: Dictionary containing paths for different profiling outputs
            parallel_setting: Parallel training setting (e.g., 'DDP')
            optimizer: Optional optimizer for backward pass tracing
            num_repeats: Number of times to repeat each operation for more accurate timing
            **kwargs: Additional arguments
        """
        super().__init__(model, model_name, output_path, parallel_setting, **kwargs)
        
        self.example_input = example_input
        self.optimizer = optimizer
        self.num_repeats = num_repeats
        
        # We have two modes of operation based on output path keys
        self.ops_profiling_path = output_path.get(PYTORCH_OPS_PROFILING, '')
        self.graph_profiling_path = output_path.get(PYTORCH_GRAPH_PROFILING, '')
        
        # Setup TorchDatabase and TorchGraph instances
        if self.ops_profiling_path:
            self.logger.info(f"Initializing Timer for {model_name} with {num_repeats} repeats")
            self.timer = Timer(num_repeats, model_name, logger=self.logger)
            self.torch_database = TorchDatabase(model, example_input, model_name, self.timer, optimizer, self.logger)
        
        if self.graph_profiling_path:
            self.torch_graph = TorchGraph(model, example_input, optimizer, model_name)
        
        self.logger.info(f"PyTorch tracer initialized for {model_name} with parallel setting: {parallel_setting}")
    
    def _get_framework_name(self) -> str:
        """Return the framework name."""
        return "PyTorch"
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete tracing process for a PyTorch model using TorchGraph and TorchDatabase.
        
        Returns:
            Dict[str, Any]: Dictionary containing the tracing results
        """
        result_data = {
            "model_name": self.model_name,
            "framework": self._get_framework_name(),
            "parallel_setting": self.parallel_setting
        }
        
        # Run ops profiling if path is specified
        if self.ops_profiling_path:
            self.logger.info(f"Running ops profiling for {self.model_name}")
            self.add_event("ops_profiling_start", {"path": self.ops_profiling_path})
            
            output_dir = self.ops_profiling_path

            # Use TorchDatabase to dump profiling results
            self.torch_database.dump_fwd_runtime(os.path.join(output_dir, 'forward_ops_profiling.json'))
            self.logger.info("Ops profiling: forward runtime completed...")
            
            if self.optimizer:
                self.torch_database.dump_bwd_runtime(os.path.join(output_dir, 'backward_ops_profiling.json'))
                self.logger.info("Ops profiling: backward runtime completed...")
            
            self.torch_database.dump_runtime(os.path.join(output_dir, 'global_ops_profiling.json'))
            self.logger.info("Ops profiling: global runtime completed...")
            
            self.add_event("ops_profiling_end", {"path": output_dir})
            
        # Run graph profiling if path is specified
        if self.graph_profiling_path:
            self.logger.info(f"Running graph profiling for {self.model_name}")
            self.add_event("graph_profiling_start", {"path": self.graph_profiling_path})
            
            output_dir = self.graph_profiling_path
        
            # Use TorchGraph to dump graph results
            self.torch_graph.dump_fwd_graph(os.path.join(output_dir, 'forward_graph_profiling.json'))
            self.logger.info("Graph profiling: forward graph completed...")
            
            if self.optimizer:
                self.torch_graph.dump_bwd_graph(os.path.join(output_dir, 'backward_graph_profiling.json'))
                self.logger.info("Graph profiling: backward graph completed...")
            
            self.torch_graph.dump_graph(os.path.join(output_dir, 'global_graph_profiling.json'))
            self.logger.info("Graph profiling: global graph completed...")
            
            self.add_event("graph_profiling_end", {"path": output_dir})
        
        # Display all captured events
        self.display_events()
        
        return result_data 