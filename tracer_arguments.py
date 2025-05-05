from common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    MODE_RUNTIME_PROFILING, MODE_GRAPH_PROFILING,
    MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL,
    Any
)
import argparse




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
        '--mode',
        type=str,
        choices=['runtime_profiling', 'graph_profiling'],
        default='runtime_profiling',
        help='Mode for PyTorch workload tracing'
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
        choices=['huggingface', 'local'], 
        default='local', 
        help='Model source'
    )
    pytorch_group.add_argument(
        '--path', 
        type=str, 
        default='output/pytorch/workload_runtime', 
        help='Output path'
    )
    pytorch_group.add_argument(
        '--batchsize', 
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


def _set_deepspeed_args(
    parser: argparse.ArgumentParser
) -> None:
    """
    Adds DeepSpeed-specific arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to.
    """
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
        '--deepspeed_batchsize', 
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
        '--megatron_batchsize', 
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
        if args.mode == 'runtime_profiling':
            filtered_dict = {k: v for k, v in vars(args).items() if k in ['model', 'path', 'batchsize', 'num_repeats', 'model_source']}
        else:
            filtered_dict = {k: v for k, v in vars(args).items() if k in ['model', 'path', 'batchsize', 'model_source']}
    elif args.framework == FRAME_NAME_DEEPSPEED:
        filtered_dict = {k.replace('deepspeed_', ''): v for k, v in vars(args).items() if k.startswith('deepspeed_')}
    elif args.framework == FRAME_NAME_MEGATRON:
        filtered_dict = {k.replace('megatron_', ''): v for k, v in vars(args).items() if k.startswith('megatron_')}
    else:
        filtered_dict = {}

    return ArgsObject(**filtered_dict)