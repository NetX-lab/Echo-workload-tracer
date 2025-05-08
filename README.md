# Echo Workload Tracer

Echo Workload Tracer is a tool designed for tracing and analyzing the workloads of different deep learning frameworks. Currently, it supports **PyTorch**, **DeepSpeed** and **Megatron-LM** frameworks.



## Project Overview
The Echo Workload Tracer focuses on capturing runtime information and generating detailed workload graphs from deep learning training jobs, using only 1 GPU device. This data is essential for analyzing performance, optimizing resource utilization, and simulating distributed training at scale.

- **PyTorch Support**: Comprehensive tracing for PyTorch models, including:
  - Support for HuggingFace Transformers models
  - Support for libs (e.g., transformers, torchvision) models and custom PyTorch models
  - Support training parallel mode like DDP
  - Capturing both forward and backward passes and extracting execution graphs and runtime data

>  **Note**：We are developing PyTorch-tracer to support PyTorch native training framework torchtitan. Tracers of Deepspeed and Megatron-LM are under active development. We would keep updating these. 


## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (at least 1 GPU device)

### Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/fwyc0573/Echo-workload-tracer.git
    cd Echo-workload-tracer
    
    export PYTHONPATH=$PYTHONPATH:/path/to/Echo-workload-tracer
    ```

2. **Setup Conda environment**
    ```bash
    conda env create -f environment.yml
    conda activate simulator_echo
    ```

## Usage

### PyTorch Tracing

#### Using the Example Script

```bash
# Basic usage (local model)
./pytorch_tracing_run.sh --model gpt2 --batch_size 32 --sequence_length 512 --num_gpus 1 --model_source local

# Advanced usage (huggingface model)
./pytorch_tracing_run_huggingface.sh --model deepseek-ai/deepseek-coder-1.3b-base --model_source huggingface --batch_size 2 --sequence_length 256 --num_repeats 5 --num_gpus 1
```

### Common Parameters

- `--framework`: Framework to use for workload tracing (choices: 'PyTorch', 'DeepSpeed', 'Megatron-LM', default: 'PyTorch')
- `--base_path`: Path to save the output (default: 'output/')
- `--num_gpus`: Number of GPUs to use in training (default: 1)

### PyTorch Parameters

- `--model`: Model to benchmark (default: 'gpt2')
- `--model_source`: Model source (choices: 'huggingface', 'local', default: 'local')
- `--batch_size`: Batch size for training/inference (default: 16)
- `--sequence_length`: Sequence length for input data (default: 512)
- `--num_repeats`: Number of repetitions for averaging results (default: 1)
- `--pytorch_ops_profiling`: Enable operations profiling for PyTorch workload
- `--pytorch_graph_profiling`: Enable graph profiling for PyTorch workload
- `--pytorch_ddp`: Enable PyTorch DistributedDataParallel (DDP) mode
- `--pytorch_only_compute_workload`: Only trace the compute workload in training

## Output

The tracer generates the following outputs:

```plaintext
output/
├── logs/
│   └── PyTorch/
│       ├── config_[model_name]_bs[batch_size]_seq[seq_length].json   # Tracing config files for each run
├── PyTorch/
│   ├── pytorch_graph_profiling/
│   │   └── [model_source]/
│   │       └── [model_name]/
│   │           ├── forward_graph_profiling_bs[batch_size]_seq[seq_length].json  # Forward graph profiling
│   │           ├── backward_graph_profiling_bs[batch_size]_seq[seq_length].json # Backward graph profiling
│   │           └── global_graph_profiling_bs[batch_size]_seq[seq_length].json   # Global graph profiling
│   └── pytorch_ops_profiling/
│       └── [model_source]/
│           └── [model_name]/
│               ├── forward_ops_profiling_bs[batch_size]_seq[seq_length].json    # Forward ops profiling
│               ├── backward_ops_profiling_bs[batch_size]_seq[seq_length].json   # Backward ops profiling
│               ├── global_ops_profiling_bs[batch_size]_seq[seq_length].json     # Global ops profiling
│               └── PyTorch tracer_[timestamp].log                               # Tracer log files
```

## Development Guide

### Adding New Model Support

To add support for additional model types:

1. Define the model loading function in the appropriate tracer file
2. Register the model source in `tracer_arguments.py`
3. Handle any model-specific operations or patterns

## Citation

If you use this tool in your research, please cite our paper:

```bibtex
@article{echo2024,
  title={Echo: Simulating Distributed Training At Scale},
  author={Yicheng Feng, Yuetao Chen, Kaiwen Chen, Jingzong Li, Tianyuan Wu, Peng Cheng, Chuan Wu, Wei Wang, Tsung-Yi Ho, Hong Xu},
  journal={arXiv preprint arXiv:2412.12487},
  year={2024}
}
```

## Contact

Please email Yicheng Feng for questions or issues related to this project.

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.
