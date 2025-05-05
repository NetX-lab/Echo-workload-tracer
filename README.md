# Echo Workload Tracer

Echo Workload Tracer 是一个用于追踪和分析不同深度学习框架工作负载的工具。目前支持 PyTorch、DeepSpeed 和 Megatron-LM 框架。

## 项目结构

```
.
├── README.md
├── tracer_arguments.py     # 命令行参数处理
├── workload_tracer.py      # 主程序入口
├── pytorch_tracing_run.sh  # PyTorch 追踪运行脚本
├── deepspeed_tracing_run.sh # DeepSpeed 追踪运行脚本
├── megatron_tracing_run.sh  # Megatron-LM 追踪运行脚本
└── output/                 # 输出目录
    ├── pytorch/           # PyTorch 输出
    ├── deepspeed/         # DeepSpeed 输出
    └── megatron/          # Megatron-LM 输出
```

## 使用方法

### 通过 Python 直接运行

```bash
# PyTorch 运行示例
python workload_tracer.py --framework PyTorch --mode runtime_profiling --model gpt2 --batchsize 32

# DeepSpeed 运行示例
python workload_tracer.py --framework DeepSpeed --deepspeed_mode inference --deepspeed_model gpt2

# Megatron-LM 运行示例
python workload_tracer.py --framework Megatron-LM --megatron_tp_size 2 --megatron_pp_size 2
```

### 通过 Shell 脚本运行

我们提供了每个框架的专用运行脚本：

```bash
# PyTorch
./pytorch_tracing_run.sh --model gpt2 --batchsize 32 --num_repeats 3

# DeepSpeed
./deepspeed_tracing_run.sh --deepspeed_model gpt2 --deepspeed_batchsize 32

# Megatron-LM
./megatron_tracing_run.sh --megatron_model gpt2 --megatron_tp_size 2 --megatron_pp_size 2
```

## 参数说明

### 通用参数

- `--framework`: 选择要使用的框架，可选值: 'PyTorch', 'DeepSpeed', 'Megatron-LM'

### PyTorch 参数

- `--mode`: 追踪模式，可选值: 'runtime_profiling', 'graph_profiling'
- `--model`: 要基准测试的模型
- `--model_source`: 模型来源，可选值: 'huggingface', 'local'
- `--path`: 输出路径
- `--batchsize`: 批处理大小
- `--num_repeats`: 重复次数

### DeepSpeed 参数

- `--deepspeed_mode`: DeepSpeed 追踪模式，可选值: 'inference', 'training'
- `--deepspeed_model`: 要测试的模型
- `--deepspeed_path`: 输出路径
- `--deepspeed_batchsize`: 批处理大小
- `--deepspeed_config`: DeepSpeed 配置文件路径

### Megatron-LM 参数

- `--megatron_mode`: Megatron 追踪模式，可选值: 'inference', 'training', 'pipeline_parallel'
- `--megatron_model`: 要测试的模型
- `--megatron_path`: 输出路径
- `--megatron_batchsize`: 批处理大小
- `--megatron_tp_size`: 张量并行度大小
- `--megatron_pp_size`: 流水线并行度大小

## 开发指南

### 添加新框架

要添加新的框架支持，请遵循以下步骤：

1. 在 `tracer_arguments.py` 中添加新的参数设置函数 `_set_<framework>_args`
2. 在 `setup_framework_args` 函数中添加新框架的条件分支
3. 在 `filter_args` 函数中添加新框架的参数过滤逻辑
4. 在 `workload_tracer.py` 中添加新的追踪函数 `run_<framework>_tracer`
5. 创建框架专用的运行脚本 `<framework>_tracing_run.sh`

## Project Overview

The Echo Workload Tracer focuses on tracing and profiling the workload of different frameworks (PyTorch, DeepSpeed, Megatron-LM) to generate detailed runtime data and workload graphs. The module is designed to support distributed training scenarios and consists of the following components:

1. **Workload Runtime**
   - Captures runtime information for LLM training workload.
   - Outputs runtime data for further analysis and simulation.

2. **Workload Graph**
   - Generates execution graphs for LLM training workload.
   - Provides insights into model structure and execution.

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (at least 1 GPU)

### Setup Instructions

1. **Clone Git Repository**
    ```bash
    git clone https://github.com/fwyc0573/Echo-workload-tracer.git
    cd Echo-workload-tracer

    export PYTHONPATH=$PYTHONPATH:/to/your/path/Echo-workload-tracer
    ```

2. **Setup Conda Environment**
    ```bash
    conda env create -f environment.yml
    conda activate echo-workload-tracer
    ```

## Usage

### Basic Execution

**Run the complete pipeline:**

```bash
sh ./tracing_run.sh
```

### Advanced Usage

You can run `tracing_run.sh` with different parameters to customize the workload tracing process. Below are the available options:
1. **Specify the framework to use**  
   Use the `--framework` parameter to specify the framework for workload tracing. Supported options include `PyTorch`, `DeepSpeed`, and `Megatron-LM`. For example:
   ```bash
   sh ./tracing_run.sh --framework PyTorch
   ```

2. **Specify the model to trace**  
   Use the `--model_source` and `--model` parameter to specify the model and the source of the model you want to trace. 
   For example, you can run a local model:
   ```bash
   sh ./tracing_run.sh --model gpt2 --model_source local
   ```
   Additionally, you can run a hugging face model:
   ```bash
   sh ./tracing_run.sh --model distilbert-base-uncased --model_source huggingface
   ```

3. **Set the output path**
    Use the `--path` parameter to set a custom directory for storing the results:
   ```bash
   sh ./tracing_run.sh --path output/pytorch/workload_runtime
   ```

4. **Set the batch size and the number of repeats**
    Use the `--batchsize` parameter to define the batch size for the workload and the `--num_repeats` parameter to specify how many times the workload should be repeated:
    ```bash
    sh ./tracing_run.sh --model gpt2 --batchsize 32 --num_repeats 5
    ```

### Parameter Summary

| Parameter       | Description                                                                 | Default Value         |
|------------------|-----------------------------------------------------------------------------|-----------------------|
| `--framework`    | Specifies the framework to use for workload tracing (`PyTorch`, `DeepSpeed`, `Megatron-LM`). | `PyTorch`            |
| `--model`        | The model to trace (e.g., `gpt2`, `bert`).                                 | `gpt2`               |
| `--model_source` | The source of the model (`huggingface` or `local`).                        | `local`        |
| `--batchsize`    | The batch size for the workload.                                           | `16`                 |
| `--num_repeats`  | The number of times to repeat the workload tracing.                        | `1`                  |
| `--path`         | The output directory for storing results.                                  | `output/pytorch/workload_runtime` |
| `--mode`         | The tracing mode (`runtime_profiling` or `graph_profiling`).               | `runtime_profiling`  |

## Output
The pipeline generates the following outputs:

```plaintext
output/
├── pytorch/
│   ├── workload_graph/
│   │   └── gpt2/                     # Example: GPT-2 workload graph
│   └── workload_runtime/             # Runtime data for workloads
```


## Expected Results

After running the pipeline, you should expect:

- Detailed workload graphs for PyTorch models.
- Runtime data for analyzing distributed training workloads.
- Validated functionality of the tracing and analysis modules.


## Citation

If you use this module in your research, please cite our paper:

```bibtex
@article{echo2024,
  title={Echo: Simulating Distributed Training At Scale},
  author={Yicheng Feng, Yuetao Chen, Kaiwen Chen, Jingzong Li, Tianyuan Wu, Peng Cheng, Chuan Wu, Wei Wang, Tsung-Yi Ho, Hong Xu},
  journal={arXiv preprint arXiv:2412.12487},
  year={2024}
}
```


## Contact

Please email Yicheng F


## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.
