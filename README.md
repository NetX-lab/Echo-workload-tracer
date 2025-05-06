# Echo Workload Tracer

Echo Workload Tracer is a tool designed for tracing and analyzing the workloads of different deep learning frameworks. Currently, it supports PyTorch, with DeepSpeed and Megatron-LM frameworks under active development.

## Overview

The Echo Workload Tracer focuses on capturing runtime information and generating detailed workload graphs from deep learning training jobs. This data is essential for analyzing performance, optimizing resource utilization, and simulating distributed training at scale.

## Project Structure

```
.
├── README.md
├── tracer_core/                # Core tracing functionality
│   ├── base.py                 # Base tracer classes
│   ├── pytorch_tracer.py       # PyTorch-specific tracing
│   ├── tracer_arguments.py     # Command-line argument handling
│   ├── tracer_initializer.py   # Tracer initialization
│   ├── torch_analysis/         # PyTorch analysis utilities
│   ├── deepspeed_tracer.py     # DeepSpeed tracing (under construction)
│   └── megatron_tracer.py      # Megatron-LM tracing (under construction)
├── workload_tracer.py          # Main program entry point
├── pytorch_tracing_run.sh      # PyTorch tracing script
└── output/                     # Output directory
```

## Features

- **PyTorch Support**: Comprehensive tracing for PyTorch models, including:
  - Support for HuggingFace Transformers models
  - Support for torchvision models
  - Support for custom PyTorch models
  - Capturing both forward and backward passes
  - Extracting execution graphs and runtime data

- **Runtime Profiling**: Collects detailed metrics on model execution
  - Operation-level timing data
  - Memory usage patterns
  - Device utilization statistics

- **Graph Profiling**: Generates visual and data representations of model execution graphs

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (at least 1 GPU device)
- Python 3.8 or higher

### Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/fwyc0573/Echo-workload-tracer.git
    cd Echo-workload-tracer
    
    export PYTHONPATH=$PYTHONPATH:/path/to/Echo-workload-tracer
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### PyTorch Tracing

#### Using the Command Line

```bash
# Basic usage
python workload_tracer.py --framework PyTorch --pytorch_ops_profiling --model gpt2 --batch_size 32

# Advanced options
python workload_tracer.py --framework PyTorch --pytorch_graph_profiling --model bert-base-uncased --model_source huggingface --batch_size 16 --num_repeats 3 --base_path output/
```

#### Using the Shell Script

```bash
# Basic usage
./pytorch_tracing_run.sh --model gpt2 --batch_size 32

# Advanced usage
./pytorch_tracing_run.sh --model resnet50 --model_source huggingface --batch_size 64 --num_repeats 5 --base_path output/
```

### PyTorch Parameters

- `--model`: Model to benchmark
- `--model_source`: Model source, options: 'huggingface', 'local'
- `--base_path`: Base output path
- `--batch_size`: Batch size
- `--num_repeats`: Number of repetitions

### DeepSpeed and Megatron-LM Support

Support for DeepSpeed and Megatron-LM frameworks is currently under construction. We are actively working on implementing comprehensive tracing capabilities for these frameworks and will update the documentation once they are available.

## Output

The tracer generates the following outputs:

```plaintext
output/
└── pytorch/
    ├── workload_graph/
    │   └── [model_name]/        # Workload graphs
    └── workload_runtime/        # Runtime data
        └── [model_name]/        # Model-specific runtime data
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
