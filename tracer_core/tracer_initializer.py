#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracer Initializer - Initialization class for creating appropriate tracers based on framework.
"""

from utils.common import (
    torch,
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PYTORCH_OPS_PROFILING, PYTORCH_GRAPH_PROFILING,PYTORCH_ONLY_COMPUTE_WORKLOAD,
    Any, Dict, Optional, Union
)
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

from utils.common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PARALLEL_SETTING_DDP
)


class TracerInitializer:
    """
    Initialization class for creating appropriate tracers based on framework.
    """
    
    @staticmethod
    def create_tracer(
        framework: str,
        model: torch.nn.Module,
        model_name: str,
        example_input: torch.Tensor,
        output_path: Dict[str, str],
        parallel_setting: Optional[str] = None,
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
                parallel_setting=parallel_setting,
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
                parallel_setting=parallel_setting,
                **kwargs
            )
        
        elif framework.lower() == FRAME_NAME_MEGATRON.lower():
            if not MEGATRON_AVAILABLE:
                raise ValueError(
                    "Megatron-LM framework specified but Megatron-LM is not installed. "
                    "Please install Megatron-LM to use the MegatronTracer."
                )
            return MegatronTracer()
        
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
    
    if args.framework == FRAME_NAME_PYTORCH or args.framework == FRAME_NAME_DEEPSPEED:

        # Extract common arguments
        framework = args.framework
        model_name = args.model
        output_path = {
            PYTORCH_OPS_PROFILING: args.ops_profiling_output_path,
            PYTORCH_GRAPH_PROFILING: args.graph_profiling_output_path
        }

        # We need to handle the model and input creation outside this function
        # as it depends on the specific model source and framework
        model = args.model_instance if hasattr(args, 'model_instance') else None
        example_input = args.example_input if hasattr(args, 'example_input') else None
        # Note: for now, we only support DDP for PyTorch
        parallel_setting = PARALLEL_SETTING_DDP if args.pytorch_ddp else None

        # Framework-specific arguments
        kwargs = {}
        
        if hasattr(args, 'optimizer'):
            kwargs['optimizer'] = args.optimizer
        if hasattr(args, 'num_repeats'):
            kwargs['num_repeats'] = args.num_repeats
        if hasattr(args, PYTORCH_ONLY_COMPUTE_WORKLOAD):
            kwargs[PYTORCH_ONLY_COMPUTE_WORKLOAD] = args.pytorch_only_compute_workload
        if hasattr(args, 'num_gpus'):
            kwargs['num_gpus'] = args.num_gpus
        if hasattr(args, 'sequence_length'):
            kwargs['sequence_length'] = args.sequence_length
        if hasattr(args, 'batch_size'):
            kwargs['batch_size'] = args.batch_size
        if hasattr(args, 'bucket_cap_mb'):
            kwargs['bucket_cap_mb'] = args.bucket_cap_mb
        kwargs['gpu_type'] = setattr(args, 'gpu_type', torch.cuda.get_device_name(0))

        return TracerInitializer.create_tracer(
            framework=framework,
            model=model,
            model_name=model_name,
            example_input=example_input,
            output_path=output_path,
            parallel_setting=parallel_setting,
            **kwargs
        )