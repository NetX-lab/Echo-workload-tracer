#!/bin/bash
export CUDA_VISIBLE_DEVICES=2


# Default parameters
PROFILING_MODE_1="pytorch_ops_profiling"
PROFILING_MODE_2="pytorch_graph_profiling"
MODEL="gpt2"
MODEL_SOURCE="local"
BASE_PATH="../output/"
BATCH_SIZE=16
SEQUENCE_LENGTH=512
NUM_REPEATS=1
NUM_GPUS=1


# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_source) MODEL_SOURCE="$2"; shift ;;
        --base_path) BASE_PATH="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --sequence_length) SEQUENCE_LENGTH="$2"; shift ;;
        --num_repeats) NUM_REPEATS="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Execute workload_tracer.py with the specified parameters
python ../main.py \
    --framework "PyTorch" \
    --model "$MODEL" \
    --model_source "$MODEL_SOURCE" \
    --base_path "$BASE_PATH" \
    --batch_size "$BATCH_SIZE" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --num_repeats "$NUM_REPEATS" \
    --num_gpus "$NUM_GPUS" \
    --$PROFILING_MODE_2
    # --pytorch_only_compute_workload \