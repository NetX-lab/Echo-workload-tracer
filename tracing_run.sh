#!/bin/bash

# Set the PATH environment variable
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Default parameters
FRAMEWORK="PyTorch"
MODE="runtime_profiling"
MODEL="gpt2"
MODEL_SOURCE="local"
PATH="output/pytorch/workload_runtime"
BATCHSIZE=16
NUM_REPEATS=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --model_source) MODEL_SOURCE="$2"; shift ;;
        --path) PATH="$2"; shift ;;
        --batchsize) BATCHSIZE="$2"; shift ;;
        --num_repeats) NUM_REPEATS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Get the current working directory
CURRENT_DIR=$(pwd)

# Get the path to the Python3 executable
PYTHON_PATH=$(command -v python3)

# Check if the current working directory is in PYTHONPATH; if not, add it
if [[ ":$PYTHONPATH:" != *":$CURRENT_DIR:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$CURRENT_DIR"
fi

# Define the path to the workload_tracer.py script
WORKLOAD_TRACER_PATH="$CURRENT_DIR/workload_tracer.py"

# Execute workload_tracer.py with the specified parameters
$PYTHON_PATH "$WORKLOAD_TRACER_PATH" --framework "$FRAMEWORK" --mode "$MODE" --model "$MODEL" --model_source "$MODEL_SOURCE" --path "$PATH" --batchsize "$BATCHSIZE" --num_repeats "$NUM_REPEATS"