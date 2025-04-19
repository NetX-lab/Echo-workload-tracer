#!/bin/bash

# Default parameters
FRAMEWORK="PyTorch"
MODE="runtime_profiling"
MODEL="gpt2"
MODEL_SOURCE="local"
OUTPUT_PATH="output/pytorch/workload_runtime"
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

# Execute workload_tracer.py with the specified parameters
python workload_tracer.py --framework "$FRAMEWORK" --mode "$MODE" --model "$MODEL" --model_source "$MODEL_SOURCE" --path "$OUTPUT_PATH" --batchsize "$BATCHSIZE" --num_repeats "$NUM_REPEATS"