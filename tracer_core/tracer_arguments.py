from utils.common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PYTORCH_OPS_PROFILING, PYTORCH_GRAPH_PROFILING,
    MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL,
    Any, os, ensure_dir_exists
)
import argparse, warnings


def check_update_args(args: Any) -> None:
    """
    Check the arguments.
    """
    assert args.num_gpus > 0, "Number of GPUs must be greater than 0"

    assert args.base_path is not None, "Output path must be specified"

    if args.framework == FRAME_NAME_PYTORCH:
        if args.num_gpus > 1:
            args.pytorch_ddp = True
        else:
            args.pytorch_ddp = False
            args.pytorch_only_compute_workload = True

        if args.pytorch_only_compute_workload:
            args.num_gpus = 1
            args.pytorch_ddp = False

        # Ensure at least one profiling mode is enabled in PyTorch simulation.
        if not args.pytorch_ops_profiling and not args.pytorch_graph_profiling:
            args.pytorch_ops_profiling = True  # Default to ops profiling if none specified
            
        args.ops_profiling_output_path = None
        if args.pytorch_ops_profiling:
            args.ops_profiling_output_path = os.path.join(args.base_path, FRAME_NAME_PYTORCH, PYTORCH_OPS_PROFILING, args.model)
            ensure_dir_exists(args.ops_profiling_output_path)

        args.graph_profiling_output_path = None
        if args.pytorch_graph_profiling:
            args.graph_profiling_output_path = os.path.join(args.base_path, FRAME_NAME_PYTORCH, PYTORCH_GRAPH_PROFILING, args.model)
            ensure_dir_exists(args.graph_profiling_output_path)
        
        args.output_log_path = os.path.join(args.base_path, 'logs', FRAME_NAME_PYTORCH)
        ensure_dir_exists(args.output_log_path)



def get_parser(
) -> argparse.ArgumentParser:
    """
    Creates and returns an ArgumentParser for command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Workload Tracer Argument Parser")

    # Arguments for the tracer framework
    tracer_group = parser.add_argument_group('Tracer')
    tracer_group.add_argument(
        '--framework', 
        type=str, 
        choices=[FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON], 
        default=FRAME_NAME_PYTORCH, 
        help='Framework to use for workload tracing'
    )
    tracer_group.add_argument(
        '--base_path', 
        type=str, 
        default='output/', 
        help='Path to save the output'
    )
    tracer_group.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        help='Number of GPUs to use in training'
    )
    return parser


def setup_framework_args(
    parser: argparse.ArgumentParser,
    framework: str
) -> argparse.ArgumentParser:
    """
    Sets up framework-specific arguments based on the chosen framework.
    
    Args:
        parser: The argument parser to add arguments to.
        framework: The chosen framework ('PyTorch', 'DeepSpeed', or 'Megatron-LM').
        
    Returns:
        The updated argument parser with framework-specific arguments.
    """
    if framework == FRAME_NAME_PYTORCH:
        _set_pytorch_args(parser)
    elif framework == FRAME_NAME_DEEPSPEED:
        _set_deepspeed_args(parser)
    elif framework == FRAME_NAME_MEGATRON:
        _set_megatron_args(parser)
    
    return parser


def _set_pytorch_args(
    parser: argparse.ArgumentParser
) -> None:
    """
    Adds PyTorch-specific arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to.
    """
    pytorch_group = parser.add_argument_group('PyTorch')
    pytorch_group.add_argument(
        '--pytorch_ops_profiling',
        action='store_true',
        default=True,
        help='Enable operations profiling for PyTorch workload'
    )
    pytorch_group.add_argument(
        '--pytorch_graph_profiling',
        action='store_true',
        default=False,
        help='Enable graph profiling for PyTorch workload'
    )
    pytorch_group.add_argument(
        '--model', 
        type=str, 
        default='gpt2', 
        help='Model to benchmark'
    )
    pytorch_group.add_argument(
        '--model_source', 
        type=str, 
        choices=[MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL], 
        default=MODEL_SOURCE_LOCAL, 
        help='Model source'
    )
    # pytorch_group.add_argument(
    #     '--path', 
    #     type=str, 
    #     default='output/pytorch/workload_runtime', 
    #     help='Output path'
    # )
    pytorch_group.add_argument(
        '--batch_size', 
        type=int, 
        default=16, 
        help='Batch size'
    )
    pytorch_group.add_argument(
        '--num_repeats', 
        type=int, 
        default=1, 
        help='Number of repeats'
    )

    pytorch_group.add_argument(
        '--pytorch_ddp',
        action='store_true',
        default=False,
        help='Enable PyTorch DistributedDataParallel (DDP) if set'
    )
    pytorch_group.add_argument(
        '--pytorch_only_compute_workload',
        action='store_true',
        default=False,
        help='Only tracing the compute workload in training'
    )


def _set_deepspeed_args(
    parser: argparse.ArgumentParser
) -> None:
    """
    Adds DeepSpeed-specific arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to.
    """
    # TODO: ADD DEEPSPEED ARGS
    deepspeed_group = parser.add_argument_group('DeepSpeed')
    deepspeed_group.add_argument(
        '--deepspeed_mode',
        type=str,
        choices=['inference', 'training'],
        default='inference',
        help='Mode for DeepSpeed workload tracing'
    )
    deepspeed_group.add_argument(
        '--deepspeed_model', 
        type=str, 
        default='gpt2', 
        help='Model to benchmark with DeepSpeed'
    )
    deepspeed_group.add_argument(
        '--deepspeed_path', 
        type=str, 
        default='output/deepspeed/workload', 
        help='Output path for DeepSpeed results'
    )
    deepspeed_group.add_argument(
        '--deepspeed_batch_size', 
        type=int, 
        default=16, 
        help='Batch size for DeepSpeed'
    )
    deepspeed_group.add_argument(
        '--deepspeed_config', 
        type=str, 
        default='configs/deepspeed_config.json', 
        help='DeepSpeed configuration file'
    )


def _set_megatron_args(
    parser: argparse.ArgumentParser
) -> None:
    """
    Adds Megatron-LM-specific arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to.
    """
    # TODO: ADD MEGATRON ARGS
    megatron_group = parser.add_argument_group('Megatron-LM')
    megatron_group.add_argument(
        '--megatron_mode',
        type=str,
        choices=['inference', 'training', 'pipeline_parallel'],
        default='inference',
        help='Mode for Megatron-LM workload tracing'
    )
    megatron_group.add_argument(
        '--megatron_model', 
        type=str, 
        default='gpt2', 
        help='Model to benchmark with Megatron-LM'
    )
    megatron_group.add_argument(
        '--megatron_path', 
        type=str, 
        default='output/megatron/workload', 
        help='Output path for Megatron-LM results'
    )
    megatron_group.add_argument(
        '--megatron_batch_size', 
        type=int, 
        default=16, 
        help='Batch size for Megatron-LM'
    )
    megatron_group.add_argument(
        '--megatron_tp_size', 
        type=int, 
        default=1, 
        help='Tensor parallelism size for Megatron-LM'
    )
    megatron_group.add_argument(
        '--megatron_pp_size', 
        type=int, 
        default=1, 
        help='Pipeline parallelism size for Megatron-LM'
    )


class ArgsObject:
    """
    A simple class to store and access arguments as attributes.
    """
    def __init__(
        self, **entries
    ) -> None:
        self.__dict__.update(entries)


def filter_args(
    args
) -> ArgsObject:
    """
    Filters arguments based on the selected framework.
    
    Args:
        args: The parsed command-line arguments.
        
    Returns:
        An ArgsObject containing only the relevant arguments for the selected framework.
    """
    if args.framework == FRAME_NAME_PYTORCH:
        # Include both profiling flags, and determine which profiling to do at runtime
        filtered_dict = {k: v for k, v in vars(args).items() 
                         if k in ['model', 'path', 'batch_size', 'num_repeats', 
                                  'model_source', 'pytorch_ops_profiling', 
                                  'pytorch_graph_profiling']}
    elif args.framework == FRAME_NAME_DEEPSPEED:
        filtered_dict = {k.replace('deepspeed_', ''): v for k, v in vars(args).items() if k.startswith('deepspeed_')}
    elif args.framework == FRAME_NAME_MEGATRON:
        filtered_dict = {k.replace('megatron_', ''): v for k, v in vars(args).items() if k.startswith('megatron_')}
    else:
        filtered_dict = {}

    return ArgsObject(**filtered_dict)