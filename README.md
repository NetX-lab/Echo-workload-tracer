# Echo Workload Tracer

This repository contains the workload tracer part of [Echo: Simulating Distributed Training at Scale](https://arxiv.org/abs/2412.12487). This part focus on the tracing and profiling work.

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
