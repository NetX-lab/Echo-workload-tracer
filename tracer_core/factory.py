#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracer Factory - Factory class for creating appropriate tracers based on framework.
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from tracer_core.base import BaseTracer
from tracer_core.pytorch_tracer import PyTorchTracer

# Import conditionally if available
try:
    from tracer_core.deepspeed_tracer import DeepSpeedTracer
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from tracer_core.megatron_tracer import MegatronTracer
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

from common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    MODE_RUNTIME_PROFILING, MODE_GRAPH_PROFILING
)


class TracerFactory:
    """
    Factory class for creating appropriate tracers based on framework.
    """
    
    @staticmethod
    def create_tracer(
        framework: str,
        model: nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: str,
        mode: str,
        **kwargs
    ) -> BaseTracer:
        """
        Create and return an appropriate tracer based on the specified framework.
        
        Args:
            framework: Framework name ('pytorch', 'deepspeed', or 'megatron')
            model: The model to trace
            model_name: Name of the model
            example_input: Example input tensor for the model
            output_path: Directory to save tracing results
            mode: Tracing mode ('runtime_profiling' or 'graph_profiling')
            **kwargs: Additional framework-specific arguments
            
        Returns:
            An appropriate tracer instance for the specified framework
            
        Raises:
            ValueError: If the framework is not supported or required modules are not available
        """
        if framework.lower() == FRAME_NAME_PYTORCH.lower():
            return PyTorchTracer(
                model=model,
                model_name=model_name,
                example_input=example_input,
                output_path=output_path,
                mode=mode,
                **kwargs
            )
        
        elif framework.lower() == FRAME_NAME_DEEPSPEED.lower():
            if not DEEPSPEED_AVAILABLE:
                raise ValueError(
                    "DeepSpeed framework specified but DeepSpeed is not installed. "
                    "Please install DeepSpeed to use the DeepSpeedTracer."
                )
            return DeepSpeedTracer(
                model=model,
                model_name=model_name,
                example_input=example_input,
                output_path=output_path,
                mode=mode,
                **kwargs
            )
        
        elif framework.lower() == FRAME_NAME_MEGATRON.lower():
            if not MEGATRON_AVAILABLE:
                raise ValueError(
                    "Megatron-LM framework specified but Megatron-LM is not installed. "
                    "Please install Megatron-LM to use the MegatronTracer."
                )
            return MegatronTracer(
                model=model,
                model_name=model_name,
                example_input=example_input,
                output_path=output_path,
                mode=mode,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported framework: {framework}. "
                           f"Supported frameworks are: {FRAME_NAME_PYTORCH}, "
                           f"{FRAME_NAME_DEEPSPEED}, and {FRAME_NAME_MEGATRON}")


def create_tracer(args: Any) -> BaseTracer:
    """
    Create a tracer based on command-line arguments.
    
    This function simplifies tracer creation when working with
    parsed command-line arguments.
    
    Args:
        args: Parsed command-line arguments containing framework, model, and other settings
        
    Returns:
        An appropriate tracer instance for the specified framework
    """
    # Extract common arguments
    framework = args.framework
    model_name = args.model
    output_path = args.path
    mode = args.mode
    
    # We need to handle the model and input creation outside this function
    # as it depends on the specific model source and framework
    model = args.model_instance if hasattr(args, 'model_instance') else None
    example_input = args.example_input if hasattr(args, 'example_input') else None
    
    # Framework-specific arguments
    kwargs = {}
    
    # Add PyTorch-specific arguments
    if framework.lower() == FRAME_NAME_PYTORCH.lower():
        if hasattr(args, 'optimizer'):
            kwargs['optimizer'] = args.optimizer
        if hasattr(args, 'num_repeats'):
            kwargs['num_repeats'] = args.num_repeats
    
    # Add DeepSpeed-specific arguments
    elif framework.lower() == FRAME_NAME_DEEPSPEED.lower():
        if hasattr(args, 'ds_config'):
            kwargs['ds_config'] = args.ds_config
        if hasattr(args, 'local_rank'):
            kwargs['local_rank'] = args.local_rank
    
    # Add Megatron-specific arguments
    elif framework.lower() == FRAME_NAME_MEGATRON.lower():
        if hasattr(args, 'tp_size'):
            kwargs['tp_size'] = args.tp_size
        if hasattr(args, 'pp_size'):
            kwargs['pp_size'] = args.pp_size
        if hasattr(args, 'micro_batch_size'):
            kwargs['micro_batch_size'] = args.micro_batch_size
        if hasattr(args, 'global_batch_size'):
            kwargs['global_batch_size'] = args.global_batch_size
    
    return TracerFactory.create_tracer(
        framework=framework,
        model=model,
        model_name=model_name,
        example_input=example_input,
        output_path=output_path,
        mode=mode,
        **kwargs
    ) 