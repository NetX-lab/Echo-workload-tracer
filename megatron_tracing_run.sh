#!/bin/bash

# Default parameters for Megatron-LM
FRAMEWORK="Megatron-LM"
MEGATRON_MODE="inference"
MEGATRON_MODEL="gpt2"
MEGATRON_PATH="output/megatron/workload"
MEGATRON_BATCHSIZE=16
MEGATRON_TP_SIZE=1
MEGATRON_PP_SIZE=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift ;;
        --megatron_mode) MEGATRON_MODE="$2"; shift ;;
        --megatron_model) MEGATRON_MODEL="$2"; shift ;;
        --megatron_path) MEGATRON_PATH="$2"; shift ;;
        --megatron_batchsize) MEGATRON_BATCHSIZE="$2"; shift ;;
        --megatron_tp_size) MEGATRON_TP_SIZE="$2"; shift ;;
        --megatron_pp_size) MEGATRON_PP_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute workload_tracer.py with the specified parameters
python workload_tracer.py \
    --framework "$FRAMEWORK" \
    --megatron_mode "$MEGATRON_MODE" \
    --megatron_model "$MEGATRON_MODEL" \
    --megatron_path "$MEGATRON_PATH" \
    --megatron_batchsize "$MEGATRON_BATCHSIZE" \
    --megatron_tp_size "$MEGATRON_TP_SIZE" \
    --megatron_pp_size "$MEGATRON_PP_SIZE"

echo "Tracing completed for $FRAMEWORK framework." 