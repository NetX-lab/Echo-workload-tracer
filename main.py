#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Workload Tracer - Main module for tracing workloads across different frameworks.
"""

from common import (
    os, sys, torch, logging, optim, 
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    MODE_RUNTIME_PROFILING, MODE_GRAPH_PROFILING,
    MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL,
    Any, ensure_dir_exists, AutoModel, AutoTokenizer
)

from tracer_arguments import get_parser, setup_framework_args, filter_args
from tracer_core.factory import create_tracer
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


def prepare_model_and_inputs(args: Any) -> None:
    """
    Prepare the model and example inputs based on arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(args.path)
    
    # Load model and prepare input tensor
    if args.model_source == MODEL_SOURCE_HUGGINGFACE:
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:  # Local model
        model = getattr(utils.transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()
    
    # Create optimizer if needed
    if args.mode == MODE_RUNTIME_PROFILING:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None
    
    # Attach to args for easy access
    setattr(args, 'model_instance', model)
    setattr(args, 'example_input', example_input)
    setattr(args, 'optimizer', optimizer)


def run_tracer(args: Any) -> None:
    """
    Run the tracer with the prepared arguments.
    """
    # Prepare the model and inputs
    prepare_model_and_inputs(args)
    
    # Create the appropriate tracer
    tracer = create_tracer(args)
    
    # Run the tracer
    result = tracer.run()
    
    # Clean up resources
    tracer.cleanup()
    
    print(f"Tracing completed for {args.model} with {args.framework} in {args.mode} mode")
    print(f"Results saved to {args.path}")


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
    
    # Display the configuration
    display_config(args)
    
    return args


def main() -> None:
    """
    Main function for the workload tracer.
    """
    # 1. Parse command-line arguments
    args = initial_args()
    
    # 2. Run the tracer with the parsed arguments
    run_tracer(args)


if __name__ == "__main__":
    main()