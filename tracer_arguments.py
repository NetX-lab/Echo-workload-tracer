import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Workload Tracer Argument Parser")

    # Arguments for the tracer framework
    tracer_group = parser.add_argument_group('Tracer')
    tracer_group.add_argument('--framework', type=str, choices=['Pytorch', 'DeepSpeed', 'Megatron-LM'], default='Pytorch', help='Framework to use for workload tracing')

    # Arguments for PyTorch
    pytorch_group = parser.add_argument_group('PyTorch')
    pytorch_group.add_argument('--mode', type=str, choices=['runtime_profiling', 'graph_profiling'], default='runtime_profiling', help='Mode for PyTorch workload tracing')
    pytorch_group.add_argument('--model', type=str, default='gpt2', help='Model to benchmark')
    pytorch_group.add_argument('--path', type=str, default='output/pytorch/workload_runtime', help='Path')
    pytorch_group.add_argument('--batchsize', type=int, default=16, help='Batch size')
    pytorch_group.add_argument('--num_repeats', type=int, default=1, help='Number of repeats')

    # Arguments for DeepSpeed
    deepspeed_group = parser.add_argument_group('DeepSpeed')
    # Arguments for Megatron-LM
    megatron_group = parser.add_argument_group('Megatron-LM')

    return parser

class ArgsObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def filter_args(args):
    if args.framework == 'Pytorch':
        if args.mode == 'runtime_profiling':
            filtered_dict = {k: v for k, v in vars(args).items() if k in ['model', 'path', 'batchsize', 'num_repeats']}
        else:
            filtered_dict = {k: v for k, v in vars(args).items() if k in ['model', 'path', 'batchsize']}
    elif args.framework == 'DeepSpeed':
        filtered_dict = {k: v for k, v in vars(args).items() if k.startswith('deepspeed_')}
    elif args.framework == 'Megatron-LM':
        filtered_dict = {k: v for k, v in vars(args).items() if k.startswith('megatron_')}
    else:
        filtered_dict = {}

    return ArgsObject(**filtered_dict)