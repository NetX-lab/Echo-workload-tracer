#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSpeed Tracer - Implementation of the tracer for DeepSpeed models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import os
import json
from collections import defaultdict

from tracer_core.base import BaseTracer

# Import DeepSpeed only if available
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


class DeepSpeedTracer(BaseTracer):
    """
    Tracer implementation for DeepSpeed-accelerated models.
    
    This tracer captures runtime and graph information for models
    running with DeepSpeed optimization, including ZeRO stages and 
    pipeline parallelism.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: str,
        mode: str,
        ds_config: Optional[Dict[str, Any]] = None,
        local_rank: int = 0,
        **kwargs
    ) -> None:
        """
        Initialize the DeepSpeed tracer.
        
        Args:
            model: PyTorch model to trace
            model_name: Name of the model
            example_input: Example input tensor for the model
            output_path: Directory to save tracing results
            mode: Tracing mode ('runtime_profiling' or 'graph_profiling')
            ds_config: DeepSpeed configuration dictionary
            local_rank: Local process rank for distributed training
            **kwargs: Additional arguments
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not installed. Please install it to use the DeepSpeedTracer.")
        
        super().__init__(model, model_name, output_path, mode, **kwargs)
        
        self.example_input = example_input
        self.ds_config = ds_config or {}
        self.local_rank = local_rank
        
        # Initialize DeepSpeed engine
        self._initialize_deepspeed()
        
        # Initialize module mapping to keep track of module execution order
        self.module_forward_order = []
        self.module_backward_order = []
        self.module_runtime_stats = defaultdict(dict)
        
        # Setup hooks for tracing
        self.setup_hooks()
        
        self.logger.info(f"DeepSpeed tracer initialized for {model_name} in {mode} mode")
    
    def _get_framework_name(self) -> str:
        """Return the framework name."""
        return "DeepSpeed"
    
    def _initialize_deepspeed(self) -> None:
        """Initialize the DeepSpeed engine with the model."""
        # Add default DeepSpeed configuration if not provided
        if not self.ds_config:
            self.ds_config = {
                "train_batch_size": 1,
                "steps_per_print": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 0.001
                    }
                },
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 1
                }
            }
        
        # Initialize DeepSpeed engine
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=parameters,
            config=self.ds_config,
            dist_init_required=False  # Set to True for multi-node distributed training
        )
        
        self.model_engine = model_engine
        self.optimizer = optimizer
        
        self.logger.info(f"DeepSpeed initialized with ZeRO stage: {self.ds_config.get('zero_optimization', {}).get('stage', 0)}")
    
    def setup_hooks(self) -> None:
        """Set up hooks for tracing DeepSpeed model execution."""
        # Register hooks for all modules
        self.forward_hooks = []
        self.backward_hooks = []
        
        # Register forward hooks
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
                
                self.module_forward_order.append(module_name)
                return None
            
            # Register hooks
            h1 = module.register_forward_pre_hook(forward_pre_hook)
            h2 = module.register_forward_hook(forward_hook)
            self.forward_hooks.extend([h1, h2])
        
        # For DeepSpeed, backward hooks are more complex due to ZeRO and gradient accumulation
        # We'll focus on capturing the overall backward pass timing
        
        self.logger.info("Set up tracing hooks for DeepSpeed model")
    
    def trace_forward(self, input_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Trace the forward pass of the DeepSpeed model.
        
        Args:
            input_data: Input tensor for the model. If None, the example_input will be used.
            
        Returns:
            torch.Tensor: Output of the model's forward pass
        """
        input_data = input_data if input_data is not None else self.example_input
        
        self.logger.info("Starting forward pass tracing")
        self.add_event("forward_start", {"input_shape": list(input_data.shape)})
        
        # Clear module order list for a new trace
        self.module_forward_order = []
        
        # Runtime profiling mode
        if self.mode == "runtime_profiling":
            start_time = time.time()
            output = self.model_engine(input_data)
            elapsed = time.time() - start_time
            self.forward_total_time = elapsed
            self.logger.info(f"Forward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            output = self.model_engine(input_data)
            self.logger.info("Forward pass graph captured")
        
        self.add_event("forward_end", {
            "output_shape": list(output.shape) if hasattr(output, "shape") else None,
            "module_order": self.module_forward_order
        })
        
        return output
    
    def trace_backward(self, loss: torch.Tensor) -> None:
        """
        Trace the backward pass of the DeepSpeed model.
        
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
            # DeepSpeed handles backward differently than PyTorch
            self.model_engine.backward(loss)
            self.model_engine.step()
            elapsed = time.time() - start_time
            self.backward_total_time = elapsed
            self.logger.info(f"Backward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            self.model_engine.backward(loss)
            self.model_engine.step()
            self.logger.info("Backward pass graph captured")
        
        self.add_event("backward_end", {"module_order": self.module_backward_order})
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete tracing process for a DeepSpeed model.
        
        Returns:
            Dict[str, Any]: Dictionary containing the tracing results
        """
        self.logger.info(f"Running {self.mode} for {self.model_name} with DeepSpeed")
        
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
            "deepspeed_config": {
                "zero_stage": self.ds_config.get("zero_optimization", {}).get("stage", 0),
                "fp16_enabled": self.ds_config.get("fp16", {}).get("enabled", False)
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
                "forward_order": self.module_forward_order,
                "backward_order": self.module_backward_order
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
                       for name, stats in self.module_runtime_stats.items()}
        }
        
        # Only save from rank 0 to avoid file conflicts
        if self.local_rank == 0:
            with open(os.path.join(output_dir, 'fwd_runtime.json'), 'w') as f:
                json.dump(fwd_runtime, f, indent=2)
        
            # Save backward runtime data if available
            if hasattr(self, "backward_total_time"):
                bwd_runtime = {
                    "total": self.backward_total_time
                }
                with open(os.path.join(output_dir, 'bwd_runtime.json'), 'w') as f:
                    json.dump(bwd_runtime, f, indent=2)
            
            # Save global runtime data
            global_runtime = {
                "forward": fwd_runtime,
                "backward": {"total": getattr(self, "backward_total_time", 0)}
            }
            with open(os.path.join(output_dir, 'global_runtime.json'), 'w') as f:
                json.dump(global_runtime, f, indent=2)
            
            self.logger.info(f"Saved runtime profiling data to {output_dir}")
    
    def _save_graph_data(self) -> None:
        """Save the graph profiling data to output files."""
        output_dir = os.path.join(self.output_path, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Only save from rank 0 to avoid file conflicts
        if self.local_rank == 0:
            # Save forward graph data
            fwd_graph = {
                "module_order": self.module_forward_order,
                "module_count": len(self.module_forward_order)
            }
            with open(os.path.join(output_dir, 'fwd_graph2.json'), 'w') as f:
                json.dump(fwd_graph, f, indent=2)
            
            # Save backward graph data
            bwd_graph = {
                "module_order": self.module_backward_order,
                "module_count": len(self.module_backward_order)
            }
            with open(os.path.join(output_dir, 'bwd_graph2.json'), 'w') as f:
                json.dump(bwd_graph, f, indent=2)
            
            # Save global graph data
            global_graph = {
                "forward": fwd_graph,
                "backward": bwd_graph
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