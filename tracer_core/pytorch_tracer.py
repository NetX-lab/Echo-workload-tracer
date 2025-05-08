#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Tracer - Implementation of the tracer for PyTorch models.
"""
from tracer_core import BaseTracer
from tracer_core.torch_analysis import Timer, TorchGraph, TorchDatabase, DDPGraph, DDPTorchDatabase
from collections import defaultdict
from utils.common import (
    torch, os, json,
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PYTORCH_OPS_PROFILING, PYTORCH_GRAPH_PROFILING, PYTORCH_ONLY_COMPUTE_WORKLOAD, PARALLEL_SETTING_DDP,
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
        local_rank: Optional[int] = 0,
        bucket_cap_mb: Optional[int] = 25,
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
            local_rank: Local rank for distributed training (default: 0)
            bucket_cap_mb: DDP bucket size in MB (default: 25)
            **kwargs: Additional arguments that can include:
                batch_size: Batch size used for training
                sequence_length: Sequence length of input data
        """
        super().__init__(model, model_name, output_path, parallel_setting, **kwargs)
        
        self.example_input = example_input
        self.optimizer = optimizer
        self.num_repeats = num_repeats
        self.local_rank = local_rank
        self.bucket_cap_mb = bucket_cap_mb
        
        # Store batch size and sequence length if provided
        self.batch_size = kwargs.get('batch_size', 0)
        self.sequence_length = kwargs.get('sequence_length', 0)
        
        # We have two modes of operation based on output path keys
        self.ops_profiling_path = output_path.get(PYTORCH_OPS_PROFILING, '')
        self.graph_profiling_path = output_path.get(PYTORCH_GRAPH_PROFILING, '')
        
        # Setup TorchDatabase and TorchGraph instances
        if self.ops_profiling_path:
            self.logger.info(f"Initializing Timer for {model_name} with {num_repeats} repeats")
            self.timer = Timer(num_repeats, model_name, logger=self.logger)
            if parallel_setting == PARALLEL_SETTING_DDP:
                self.logger.info(f"Using DDPTorchDatabase for {model_name}")
                self.torch_database = DDPTorchDatabase(
                    model, 
                    example_input, 
                    model_name, 
                    self.timer, 
                    optimizer, 
                    self.logger
                )
            else:
                self.torch_database = TorchDatabase(
                    model, 
                    example_input, 
                    model_name, 
                    self.timer, 
                    optimizer, 
                    self.logger
                )
        
        if self.graph_profiling_path:
            self.logger.info(f"Initializing TorchGraph for {model_name}")
            if parallel_setting == PARALLEL_SETTING_DDP:
                self.torch_graph = DDPGraph(
                    model, 
                    example_input, 
                    optimizer, 
                    model_name, 
                    self.local_rank,
                    bucket_cap_mb=self.bucket_cap_mb,
                    logger=self.logger
                )
            else:
                self.torch_graph = TorchGraph(model, example_input, optimizer, model_name, logger=self.logger)
        
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
        
        # Generate suffix for output filenames
        suffix = self._generate_filename_suffix()
        
        # Run ops profiling if path is specified
        if self.ops_profiling_path:
            self._run_ops_profiling(suffix)
        
        # Run graph profiling if path is specified
        if self.graph_profiling_path:
            self._run_graph_profiling(suffix)
        
        # Display all captured events
        self.display_events()
        
        return result_data

    def _generate_filename_suffix(self) -> str:
        """
        Generate a suffix string for output filenames based on batch size, 
        sequence length, and parallel setting.
        
        Returns:
            str: The generated suffix
        """
        batch_size = getattr(self, 'batch_size', 0)
        seq_length = getattr(self, 'sequence_length', 0)
        suffix = f"_bs{batch_size}_seq{seq_length}"
        
        if self.parallel_setting is not None:
            suffix += f"_{self.parallel_setting}"
        
        return suffix

    def _run_ops_profiling(self, suffix: str) -> None:
        """
        Run operations profiling and save the results.
        
        Args:
            suffix: Suffix to append to output filenames
        """
        self.logger.info(f"Running ops profiling for {self.model_name}")
        self.add_event("ops_profiling_start", {"path": self.ops_profiling_path})
        
        output_dir = self.ops_profiling_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output files with appropriate suffixes
        forward_ops_file = os.path.join(output_dir, f'forward_ops_profiling{suffix}.json')
        backward_ops_file = os.path.join(output_dir, f'backward_ops_profiling{suffix}.json')
        global_ops_file = os.path.join(output_dir, f'global_ops_profiling{suffix}.json')
        
        # Save forward profiling data
        self.torch_database.dump_fwd_runtime(forward_ops_file)
        self.logger.info(f"Ops profiling: forward runtime saved to {forward_ops_file}")
        
        # Save backward profiling data if optimizer is available
        if self.optimizer:
            self.torch_database.dump_bwd_runtime(backward_ops_file)
            self.logger.info(f"Ops profiling: backward runtime saved to {backward_ops_file}")
        
        # Save global profiling data
        self.torch_database.dump_runtime(global_ops_file)
        self.logger.info(f"Ops profiling: global runtime saved to {global_ops_file}")
        
        self.add_event("ops_profiling_end", {"path": output_dir})

    def _run_graph_profiling(self, suffix: str) -> None:
        """
        Run graph profiling and save the results.
        
        Args:
            suffix: Suffix to append to output filenames
        """
        self.logger.info(f"Running graph profiling for {self.model_name}")
        self.add_event("graph_profiling_start", {"path": self.graph_profiling_path})
        
        output_dir = self.graph_profiling_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output files with appropriate suffixes
        forward_graph_file = os.path.join(output_dir, f'forward_graph_profiling{suffix}.json')
        backward_graph_file = os.path.join(output_dir, f'backward_graph_profiling{suffix}.json')
        global_graph_file = os.path.join(output_dir, f'global_graph_profiling{suffix}.json')
        
        # Save forward graph data
        self.torch_graph.dump_fwd_graph(forward_graph_file)
        self.logger.info(f"Graph profiling: forward graph saved to {forward_graph_file}")
        
        # Save backward graph data if optimizer is available
        if self.optimizer:
            self.torch_graph.dump_bwd_graph(backward_graph_file)
            self.logger.info(f"Graph profiling: backward graph saved to {backward_graph_file}")
        
        # Save global graph data
        self.torch_graph.dump_graph(global_graph_file)
        self.logger.info(f"Graph profiling: global graph saved to {global_graph_file}")
        
        # If using DDP, save additional DDP specific data
        if self.parallel_setting == PARALLEL_SETTING_DDP and hasattr(self.torch_graph, 'dump_ddp_graph'):
            ddp_graph_file = os.path.join(output_dir, f'ddp_graph_profiling{suffix}.json')
            self.torch_graph.dump_ddp_graph(ddp_graph_file)
            self.logger.info(f"Graph profiling: DDP graph saved to {ddp_graph_file}")
        
        self.add_event("graph_profiling_end", {"path": output_dir}) 