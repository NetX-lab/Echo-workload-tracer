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

from tracer_core.tracer_arguments import get_parser, setup_framework_args, check_args
from tracer_core import create_tracer
from utils import (
    prepare_model_and_inputs,
    display_config
)

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
    # Get basic parser with framework selection
    parser = get_parser()
    
    # Parse initial args to get framework (parse_known_args ignores unknown args)
    initial_args, _ = parser.parse_known_args()
    
    # Setup framework-specific args
    parser = setup_framework_args(parser, initial_args.framework)
    
    # Parse all args
    args = parser.parse_args()
    
    # Check args
    check_args(args)


    # Display the configuration
    display_config(args)
    
    return args

def main() -> None:
    """
    Main function for the workload tracer.
    """
    # Parse command-line arguments
    args = initial_args()
    
    # Run the tracer with the parsed arguments
    run_tracer(args)

if __name__ == "__main__":
    main()