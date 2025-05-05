#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Megatron-LM Tracer - Implementation of the tracer for Megatron-LM models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import os
import json
from collections import defaultdict

from tracer_core.base import BaseTracer

# Import Megatron-LM only if available
try:
    import megatron
    from megatron import get_args
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False


class MegatronTracer(BaseTracer):
    """
    Tracer implementation for Megatron-LM models.
    
    This tracer captures runtime and graph information for models
    running with Megatron-LM's tensor and pipeline parallelism.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: str,
        mode: str,
        tp_size: int = 1,
        pp_size: int = 1,
        micro_batch_size: int = 1,
        global_batch_size: int = 1,
        **kwargs
    ) -> None:
        """
        Initialize the Megatron-LM tracer.
        
        Args:
            model: PyTorch model to trace
            model_name: Name of the model
            example_input: Example input tensor for the model
            output_path: Directory to save tracing results
            mode: Tracing mode ('runtime_profiling' or 'graph_profiling')
            tp_size: Tensor parallelism size
            pp_size: Pipeline parallelism size
            micro_batch_size: Micro batch size for pipeline parallelism
            global_batch_size: Global batch size for training
            **kwargs: Additional arguments
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron-LM is not installed. Please install it to use the MegatronTracer.")
        
        super().__init__(model, model_name, output_path, mode, **kwargs)
        
        self.example_input = example_input
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        
        # Get Megatron-LM arguments
        self.megatron_args = get_args()
        
        # Initialize module mapping to keep track of module execution order
        self.module_forward_order = []
        self.module_backward_order = []
        self.module_runtime_stats = defaultdict(dict)
        
        # Track parallel execution statistics
        self.parallel_stats = {
            "tensor_parallel": {
                "size": tp_size,
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            },
            "pipeline_parallel": {
                "size": pp_size,
                "microbatches": global_batch_size // micro_batch_size if micro_batch_size > 0 else 1
            }
        }
        
        # Setup hooks for tracing
        self.setup_hooks()
        
        self.logger.info(f"Megatron-LM tracer initialized for {model_name} in {mode} mode")
        self.logger.info(f"Tensor Parallel Size: {tp_size}, Pipeline Parallel Size: {pp_size}")
    
    def _get_framework_name(self) -> str:
        """Return the framework name."""
        return "Megatron-LM"
    
    def setup_hooks(self) -> None:
        """Set up hooks for tracing Megatron-LM model execution."""
        # Register hooks for all modules
        self.forward_hooks = []
        self.backward_hooks = []
        
        # For Megatron-LM, we need to consider tensor and pipeline parallelism
        # which affects how modules are executed across devices
        
        # Register forward hooks for available modules
        for name, module in self.model.named_modules():
            if name == "":  # Skip the root module
                continue
                
            # Forward pre-hook to capture input
            def forward_pre_hook(module, input, module_name=name):
                if self.mode == "runtime_profiling":
                    start_time = time.time()
                    module._trace_start_time = start_time
                return None
            
            # Forward hook to capture output and execution time
            def forward_hook(module, input, output, module_name=name):
                if self.mode == "runtime_profiling" and hasattr(module, "_trace_start_time"):
                    elapsed = time.time() - module._trace_start_time
                    self.module_runtime_stats[module_name]["forward"] = elapsed
                
                # Add module info including parallel context
                self.module_forward_order.append({
                    "name": module_name,
                    "tp_rank": self.parallel_stats["tensor_parallel"]["rank"],
                    "pp_stage": getattr(module, "_pp_stage", 0)  # Attempt to get pipeline stage if available
                })
                return None
            
            # Register hooks
            h1 = module.register_forward_pre_hook(forward_pre_hook)
            h2 = module.register_forward_hook(forward_hook)
            self.forward_hooks.extend([h1, h2])
        
        # For Megatron-LM, backward hooks are complex due to parallelism strategies
        # We'll focus on capturing the overall backward pass timing
        
        self.logger.info("Set up tracing hooks for Megatron-LM model")
    
    def trace_forward(self, input_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Trace the forward pass of the Megatron-LM model.
        
        Args:
            input_data: Input tensor for the model. If None, the example_input will be used.
            
        Returns:
            torch.Tensor: Output of the model's forward pass
        """
        input_data = input_data if input_data is not None else self.example_input
        
        self.logger.info("Starting forward pass tracing")
        self.add_event("forward_start", {
            "input_shape": list(input_data.shape),
            "tp_size": self.tp_size,
            "pp_size": self.pp_size
        })
        
        # Clear module order list for a new trace
        self.module_forward_order = []
        
        # Runtime profiling mode
        if self.mode == "runtime_profiling":
            start_time = time.time()
            # Megatron-LM models might have a different interface
            # Adapt this based on your specific model implementation
            output = self.model(input_data)
            elapsed = time.time() - start_time
            self.forward_total_time = elapsed
            self.logger.info(f"Forward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            output = self.model(input_data)
            self.logger.info("Forward pass graph captured")
        
        self.add_event("forward_end", {
            "output_shape": list(output.shape) if hasattr(output, "shape") else None,
            "module_count": len(self.module_forward_order)
        })
        
        return output
    
    def trace_backward(self, loss: torch.Tensor) -> None:
        """
        Trace the backward pass of the Megatron-LM model.
        
        Args:
            loss: The loss tensor to backpropagate
        """
        self.logger.info("Starting backward pass tracing")
        self.add_event("backward_start", {"loss_value": float(loss.item())})
        
        # Clear module order list for a new trace
        self.module_backward_order = []
        
        # Runtime profiling mode
        if self.mode == "runtime_profiling":
            start_time = time.time()
            # For Megatron-LM, the backward pass may involve communication across parallel ranks
            loss.backward()
            elapsed = time.time() - start_time
            self.backward_total_time = elapsed
            self.logger.info(f"Backward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            loss.backward()
            self.logger.info("Backward pass graph captured")
        
        self.add_event("backward_end", {"module_count": len(self.module_backward_order)})
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete tracing process for a Megatron-LM model.
        
        Returns:
            Dict[str, Any]: Dictionary containing the tracing results
        """
        self.logger.info(f"Running {self.mode} for {self.model_name} with Megatron-LM")
        self.logger.info(f"Tensor Parallel Size: {self.tp_size}, Pipeline Parallel Size: {self.pp_size}")
        
        # Perform forward pass
        output = self.trace_forward()
        
        # Create a simple loss for backward pass
        if hasattr(output, "mean"):
            loss = output.mean()
            # Perform backward pass
            self.trace_backward(loss)
        
        # Process and save trace data based on mode
        result_data = self._process_trace_data()
        
        # Save the trace data
        self.save_trace()
        
        # Save mode-specific output files
        if self.mode == "runtime_profiling":
            self._save_runtime_data()
        else:  # graph_profiling
            self._save_graph_data()
        
        return result_data
    
    def _process_trace_data(self) -> Dict[str, Any]:
        """
        Process the collected trace data based on the tracing mode.
        
        Returns:
            Dict[str, Any]: Processed trace data
        """
        result_data = {
            "model_name": self.model_name,
            "mode": self.mode,
            "framework": self._get_framework_name(),
            "parallelism": {
                "tensor_parallel_size": self.tp_size,
                "pipeline_parallel_size": self.pp_size,
                "micro_batch_size": self.micro_batch_size,
                "global_batch_size": self.global_batch_size
            }
        }
        
        if self.mode == "runtime_profiling":
            # Collect runtime statistics
            runtime_stats = {
                "forward_total": getattr(self, "forward_total_time", 0),
                "module_stats": self.module_runtime_stats
            }
            
            if hasattr(self, "backward_total_time"):
                runtime_stats["backward_total"] = self.backward_total_time
            
            result_data["runtime_stats"] = runtime_stats
        else:  # graph_profiling
            # Collect graph structure
            result_data["graph"] = {
                "forward_modules": self.module_forward_order,
                "backward_modules": self.module_backward_order
            }
        
        return result_data
    
    def _save_runtime_data(self) -> None:
        """Save the runtime profiling data to output files."""
        output_dir = os.path.join(self.output_path, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forward runtime data
        fwd_runtime = {
            "total": getattr(self, "forward_total_time", 0),
            "modules": {name: stats.get("forward", 0) 
                       for name, stats in self.module_runtime_stats.items()},
            "parallelism": {
                "tp_size": self.tp_size,
                "pp_size": self.pp_size
            }
        }
        
        # Only save from one process to avoid file conflicts
        # In a real implementation, you might want to gather data from all ranks
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            with open(os.path.join(output_dir, 'fwd_runtime.json'), 'w') as f:
                json.dump(fwd_runtime, f, indent=2)
        
            # Save backward runtime data if available
            if hasattr(self, "backward_total_time"):
                bwd_runtime = {
                    "total": self.backward_total_time,
                    "parallelism": {
                        "tp_size": self.tp_size,
                        "pp_size": self.pp_size
                    }
                }
                with open(os.path.join(output_dir, 'bwd_runtime.json'), 'w') as f:
                    json.dump(bwd_runtime, f, indent=2)
            
            # Save global runtime data
            global_runtime = {
                "forward": fwd_runtime,
                "backward": {"total": getattr(self, "backward_total_time", 0)},
                "parallelism": {
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size,
                    "micro_batch_size": self.micro_batch_size,
                    "global_batch_size": self.global_batch_size
                }
            }
            with open(os.path.join(output_dir, 'global_runtime.json'), 'w') as f:
                json.dump(global_runtime, f, indent=2)
            
            self.logger.info(f"Saved runtime profiling data to {output_dir}")
    
    def _save_graph_data(self) -> None:
        """Save the graph profiling data to output files."""
        output_dir = os.path.join(self.output_path, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Only save from one process to avoid file conflicts
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            # Save forward graph data
            fwd_graph = {
                "modules": self.module_forward_order,
                "module_count": len(self.module_forward_order),
                "parallelism": {
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size
                }
            }
            with open(os.path.join(output_dir, 'fwd_graph2.json'), 'w') as f:
                json.dump(fwd_graph, f, indent=2)
            
            # Save backward graph data
            bwd_graph = {
                "modules": self.module_backward_order,
                "module_count": len(self.module_backward_order),
                "parallelism": {
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size
                }
            }
            with open(os.path.join(output_dir, 'bwd_graph2.json'), 'w') as f:
                json.dump(bwd_graph, f, indent=2)
            
            # Save global graph data
            global_graph = {
                "forward": fwd_graph,
                "backward": bwd_graph,
                "parallelism": {
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size,
                    "micro_batch_size": self.micro_batch_size,
                    "global_batch_size": self.global_batch_size
                }
            }
            with open(os.path.join(output_dir, 'global_graph2.json'), 'w') as f:
                json.dump(global_graph, f, indent=2)
            
            self.logger.info(f"Saved graph profiling data to {output_dir}")
    
    def cleanup(self) -> None:
        """Clean up hooks and resources used by the tracer."""
        # Remove forward hooks
        for hook in self.forward_hooks:
            hook.remove()
        
        # Remove backward hooks
        for hook in self.backward_hooks:
            hook.remove()
        
        self.logger.info("Cleaned up all tracing hooks") 