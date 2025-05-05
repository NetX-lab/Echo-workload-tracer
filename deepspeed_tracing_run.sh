#!/bin/bash

# Default parameters for DeepSpeed
FRAMEWORK="DeepSpeed"
DEEPSPEED_MODE="inference"
DEEPSPEED_MODEL="gpt2"
DEEPSPEED_PATH="output/deepspeed/workload"
DEEPSPEED_BATCHSIZE=16
DEEPSPEED_CONFIG="configs/deepspeed_config.json"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift ;;
        --deepspeed_mode) DEEPSPEED_MODE="$2"; shift ;;
        --deepspeed_model) DEEPSPEED_MODEL="$2"; shift ;;
        --deepspeed_path) DEEPSPEED_PATH="$2"; shift ;;
        --deepspeed_batchsize) DEEPSPEED_BATCHSIZE="$2"; shift ;;
        --deepspeed_config) DEEPSPEED_CONFIG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute workload_tracer.py with the specified parameters
python workload_tracer.py \
    --framework "$FRAMEWORK" \
    --deepspeed_mode "$DEEPSPEED_MODE" \
    --deepspeed_model "$DEEPSPEED_MODEL" \
    --deepspeed_path "$DEEPSPEED_PATH" \
    --deepspeed_batchsize "$DEEPSPEED_BATCHSIZE" \
    --deepspeed_config "$DEEPSPEED_CONFIG"

echo "Tracing completed for $FRAMEWORK framework." 