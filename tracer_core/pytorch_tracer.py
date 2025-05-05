#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Tracer - Implementation of the tracer for PyTorch models.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import os
import json
from collections import defaultdict

from tracer_core.base import BaseTracer
from torch_analysis.profiling_timer import Timer


class PyTorchTracer(BaseTracer):
    """
    Tracer implementation for PyTorch models.
    
    This tracer captures runtime and graph information for PyTorch models
    during both forward and backward passes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: str,
        mode: str,
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
            output_path: Directory to save tracing results
            mode: Tracing mode ('runtime_profiling' or 'graph_profiling')
            optimizer: Optional optimizer for backward pass tracing
            num_repeats: Number of times to repeat each operation for more accurate timing
            **kwargs: Additional arguments
        """
        super().__init__(model, model_name, output_path, mode, **kwargs)
        
        self.example_input = example_input
        self.optimizer = optimizer
        self.num_repeats = num_repeats
        
        # Create timer for runtime profiling
        if self.mode == "runtime_profiling":
            self.timer = Timer(num_repeats, model_name)
        
        # Initialize module mapping to keep track of module execution order
        self.module_forward_order = []
        self.module_backward_order = []
        self.module_runtime_stats = defaultdict(dict)
        
        # Setup hooks for tracing
        self.setup_hooks()
        
        self.logger.info(f"PyTorch tracer initialized for {model_name} in {mode} mode")
    
    def _get_framework_name(self) -> str:
        """Return the framework name."""
        return "PyTorch"
    
    def setup_hooks(self) -> None:
        """Set up hooks for tracing PyTorch model execution."""
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
                    self.timer.start(f"fwd_{module_name}")
                return None
            
            # Forward hook to capture output and execution time
            def forward_hook(module, input, output, module_name=name):
                if self.mode == "runtime_profiling":
                    elapsed = self.timer.stop(f"fwd_{module_name}")
                    self.module_runtime_stats[module_name]["forward"] = elapsed
                
                self.module_forward_order.append(module_name)
                return None
            
            # Register hooks
            h1 = module.register_forward_pre_hook(forward_pre_hook)
            h2 = module.register_forward_hook(forward_hook)
            self.forward_hooks.extend([h1, h2])
        
        # Register backward hooks if optimizer is provided
        if self.optimizer:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Backward hook to capture gradient computation
                    def backward_hook(grad, param_name=name):
                        if param_name not in self.module_backward_order:
                            self.module_backward_order.append(param_name)
                        return None
                    
                    h = param.register_hook(backward_hook)
                    self.backward_hooks.append(h)
        
        self.logger.info("Set up tracing hooks for PyTorch model")
    
    def trace_forward(self, input_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Trace the forward pass of the PyTorch model.
        
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
            self.timer.start("forward_total")
            output = self.model(input_data)
            elapsed = self.timer.stop("forward_total")
            self.logger.info(f"Forward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            output = self.model(input_data)
            self.logger.info("Forward pass graph captured")
        
        self.add_event("forward_end", {
            "output_shape": list(output.shape) if hasattr(output, "shape") else None,
            "module_order": self.module_forward_order
        })
        
        return output
    
    def trace_backward(self, loss: torch.Tensor) -> None:
        """
        Trace the backward pass of the PyTorch model.
        
        Args:
            loss: The loss tensor to backpropagate
        """
        if self.optimizer is None:
            self.logger.warning("No optimizer provided, skipping backward pass tracing")
            return
        
        self.logger.info("Starting backward pass tracing")
        self.add_event("backward_start", {"loss_value": float(loss.item())})
        
        # Clear module order list for a new trace
        self.module_backward_order = []
        
        # Runtime profiling mode
        if self.mode == "runtime_profiling":
            self.timer.start("backward_total")
            loss.backward()
            elapsed = self.timer.stop("backward_total")
            self.logger.info(f"Backward pass completed in {elapsed:.6f} seconds")
        else:  # Graph profiling mode
            loss.backward()
            self.logger.info("Backward pass graph captured")
        
        self.add_event("backward_end", {"module_order": self.module_backward_order})
        
        # Optimizer step
        if self.optimizer:
            if self.mode == "runtime_profiling":
                self.timer.start("optimizer_step")
                self.optimizer.step()
                elapsed = self.timer.stop("optimizer_step")
                self.logger.info(f"Optimizer step completed in {elapsed:.6f} seconds")
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete tracing process for a PyTorch model.
        
        Returns:
            Dict[str, Any]: Dictionary containing the tracing results
        """
        self.logger.info(f"Running {self.mode} for {self.model_name}")
        
        # Perform forward pass
        output = self.trace_forward()
        
        # Create a simple loss for backward pass
        if self.optimizer:
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
            "framework": self._get_framework_name()
        }
        
        if self.mode == "runtime_profiling":
            # Collect runtime statistics
            runtime_stats = {
                "forward_total": self.timer.get_time("forward_total"),
                "module_stats": self.module_runtime_stats
            }
            
            if self.optimizer:
                runtime_stats["backward_total"] = self.timer.get_time("backward_total")
                runtime_stats["optimizer_step"] = self.timer.get_time("optimizer_step")
            
            result_data["runtime_stats"] = runtime_stats
        else:  # graph_profiling
            # Collect graph structure
            result_data["graph"] = {
                "forward_order": self.module_forward_order,
                "backward_order": self.module_backward_order if self.optimizer else []
            }
        
        return result_data
    
    def _save_runtime_data(self) -> None:
        """Save the runtime profiling data to output files."""
        output_dir = os.path.join(self.output_path, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forward runtime data
        fwd_runtime = {
            "total": self.timer.get_time("forward_total"),
            "modules": {name: stats.get("forward", 0) 
                       for name, stats in self.module_runtime_stats.items()}
        }
        with open(os.path.join(output_dir, 'fwd_runtime.json'), 'w') as f:
            json.dump(fwd_runtime, f, indent=2)
        
        # Save backward runtime data if optimizer exists
        if self.optimizer:
            bwd_runtime = {
                "total": self.timer.get_time("backward_total"),
                "optimizer_step": self.timer.get_time("optimizer_step")
            }
            with open(os.path.join(output_dir, 'bwd_runtime.json'), 'w') as f:
                json.dump(bwd_runtime, f, indent=2)
        
        # Save global runtime data
        global_runtime = {
            "forward": fwd_runtime,
            "backward": bwd_runtime if self.optimizer else {"total": 0}
        }
        with open(os.path.join(output_dir, 'global_runtime.json'), 'w') as f:
            json.dump(global_runtime, f, indent=2)
        
        self.logger.info(f"Saved runtime profiling data to {output_dir}")
    
    def _save_graph_data(self) -> None:
        """Save the graph profiling data to output files."""
        output_dir = os.path.join(self.output_path, self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forward graph data
        fwd_graph = {
            "module_order": self.module_forward_order,
            "module_count": len(self.module_forward_order)
        }
        with open(os.path.join(output_dir, 'fwd_graph2.json'), 'w') as f:
            json.dump(fwd_graph, f, indent=2)
        
        # Save backward graph data if optimizer exists
        if self.optimizer:
            bwd_graph = {
                "module_order": self.module_backward_order,
                "module_count": len(self.module_backward_order)
            }
            with open(os.path.join(output_dir, 'bwd_graph2.json'), 'w') as f:
                json.dump(bwd_graph, f, indent=2)
        
        # Save global graph data
        global_graph = {
            "forward": fwd_graph,
            "backward": bwd_graph if self.optimizer else {"module_count": 0}
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