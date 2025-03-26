#!/bin/bash

# 设置 PATH 环境变量
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# 默认参数
FRAMEWORK="Pytorch"
MODE="runtime_profiling"
MODEL="gpt2"
PATH="output/pytorch/workload_runtime"
BATCHSIZE=16
NUM_REPEATS=1

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --path) PATH="$2"; shift ;;
        --batchsize) BATCHSIZE="$2"; shift ;;
        --num_repeats) NUM_REPEATS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 获取当前执行路径
CURRENT_DIR=$(pwd)

# 获取当前 Python3 路径
PYTHON_PATH=$(command -v python3)

# 检查当前执行路径是否在 PYTHONPATH 中，如果不在则加上
if [[ ":$PYTHONPATH:" != *":$CURRENT_DIR:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$CURRENT_DIR"
fi

WORKLOAD_TRACER_PATH="$CURRENT_DIR/workload_tracer.py"

# 运行 workload_tracer.py
$PYTHON_PATH "$WORKLOAD_TRACER_PATH" --framework "$FRAMEWORK" --mode "$MODE" --model "$MODEL" --path "$PATH" --batchsize "$BATCHSIZE" --num_repeats "$NUM_REPEATS"