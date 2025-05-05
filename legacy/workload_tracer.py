#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Workload Tracer - Main module for tracing workloads across different frameworks.
"""

from utils.common import (
    os, sys, torch, logging, optim, 
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    MODE_RUNTIME_PROFILING, MODE_GRAPH_PROFILING,
    MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL,
    Any, ensure_dir_exists, AutoModel, AutoTokenizer
)
# 项目内部导入
from tracer_core.tracer_arguments import get_parser, setup_framework_args, filter_args
from torch_analysis.torch_database import TorchDatabase
from torch_analysis.torch_graph import TorchGraph
from torch_analysis.profiling_timer import Timer
import utils.transformer
from utils.config_display import get_config_display



def load_huggingface_model(
    model_name: str
) -> tuple:
    """
    Load a Hugging Face model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_torch_database_test(
    args
) -> None:
    """
    Runs the TorchDatabase test for runtime profiling.
    """
    timer = Timer(args.num_repeats, args.model)

    if args.model_source == 'huggingface':  #  Hugging Face support
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:
        model = getattr(utils.transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    g = TorchDatabase(model, example_input, args.model, timer, optimizer)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_runtime(os.path.join(output_dir, 'fwd_runtime.json'))
    g.dump_bwd_runtime(os.path.join(output_dir, 'bwd_runtime.json'))
    g.dump_runtime(os.path.join(output_dir, 'global_runtime.json'))


def run_torch_graph_test(
    args
) -> None:
    """
    Runs the TorchGraph test for graph profiling.
    """
    if args.model_source == 'huggingface':  #  Hugging Face support
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:
        model = getattr(utils.transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    g = TorchGraph(model, example_input, optimizer, args.model)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
    logging.info("torch_graph: forward graph completed...")

    g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
    logging.info("torch_graph: backward graph completed...")

    g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
    logging.info("torch_graph: global graph completed...")


def display_config(args: Any) -> None:
    """
    Display the configuration of the workload tracer.
    """
    # Add hardware information to args if using PyTorch
    if torch.cuda.is_available():
        setattr(args, '_cuda_available', True)
        setattr(args, '_gpu_name', torch.cuda.get_device_name(0))
        setattr(args, '_gpu_memory', f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    else:
        setattr(args, '_cuda_available', False)
        raise ValueError("Echo need at least one GPU to run.")

    # Get the appropriate config display instance and display configuration
    config_display = get_config_display(args)
    config_display.display()


def run_tracer(args: Any) -> None:
    """
    Run the tracer with the filtered arguments.
    """
    if args.framework == FRAME_NAME_PYTORCH:
        run_pytorch_tracer(args)
    elif args.framework == FRAME_NAME_DEEPSPEED:
        run_deepspeed_tracer(args)
    elif args.framework == FRAME_NAME_MEGATRON:
        run_megatron_tracer(args)
    else:
        print(f"Unsupported framework: {args.framework}")
        sys.exit(1)


def initial_args() -> Any:
    """
    Get the initial arguments.
    """
    # Step 1: Get basic parser with framework selection
    parser = get_parser()
    
    # Step 2: Parse initial args to get framework (parse_known_args ignores unknown args)
    initial_args, _ = parser.parse_known_args()
    
    # Step 3: Setup framework-specific args
    parser = setup_framework_args(parser, initial_args.framework)
    
    # Step 4: Parse all args
    args = parser.parse_args()
    
    # # Step 5: Filter args for the selected framework
    # filtered_args = filter_args(args)
    
    # Display the configuration
    display_config(args)
    
    return args


def create_tracer(args: Any) -> Any:
    """
    Create a tracer based on the arguments.
    """
    return args


def main() -> None:
    """
    Main function for the workload tracer.
    """
    # 1. initial args
    args = initial_args()

    # 2. create tracer
    tracer = create_tracer(args)

    # 3. run tracer
    run_tracer(tracer, args)






def run_pytorch_tracer(args: Any) -> None:
    """
    Run the PyTorch tracer with the filtered arguments.
    
    Args:
        args: Filtered arguments for PyTorch tracer.
    """
    print("Running PyTorch tracer with arguments:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")
    
    # Placeholder for actual PyTorch tracing logic
    if hasattr(args, 'mode') and args.mode == 'runtime_profiling':
        print(f"Performing runtime profiling for model {args.model} with batch size {args.batchsize}")
        # Implement runtime profiling logic here
    else:
        print(f"Performing graph profiling for model {args.model} with batch size {args.batchsize}")
        # Implement graph profiling logic here


def run_deepspeed_tracer(args: Any) -> None:
    """
    Run the DeepSpeed tracer with the filtered arguments.
    
    Args:
        args: Filtered arguments for DeepSpeed tracer.
    """
    print("Running DeepSpeed tracer with arguments:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")
    
    # Placeholder for actual DeepSpeed tracing logic
    print(f"Tracing DeepSpeed {args.mode} workload with model {args.model}")
    # Implement DeepSpeed tracing logic here


def run_megatron_tracer(args: Any) -> None:
    """
    Run the Megatron-LM tracer with the filtered arguments.
    
    Args:
        args: Filtered arguments for Megatron-LM tracer.
    """
    print("Running Megatron-LM tracer with arguments:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {value}")
    
    # Placeholder for actual Megatron-LM tracing logic
    print(f"Tracing Megatron-LM {args.mode} workload with model {args.model}")
    print(f"Tensor Parallelism: {args.tp_size}, Pipeline Parallelism: {args.pp_size}")
    # Implement Megatron-LM tracing logic here


if __name__ == "__main__":
    main()