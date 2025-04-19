import warnings
import subprocess
import os
import sys
import torch
import json
import logging

# sys.path.append('/root/Echo-workload-tracer/torch_analysis')
from tracer_arguments import get_parser, filter_args
from torch_analysis.torch_database import TorchDatabase
from torch_analysis.torch_graph import TorchGraph
from torch_analysis.profiling_timer import Timer
import torch.optim as optim
import utils.transformer
from utils.config_display import get_config_display
from transformers import AutoModel, AutoTokenizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_huggingface_model(
    model_name: str
) -> tuple:
    """
    Load a Hugging Face model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_torch_database_test(
    args
) -> None:
    """
    Runs the TorchDatabase test for runtime profiling.
    """
    timer = Timer(args.num_repeats, args.model)

    if args.model_source == 'huggingface':  #  Hugging Face support
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:
        model = getattr(utils.transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    g = TorchDatabase(model, example_input, args.model, timer, optimizer)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_runtime(os.path.join(output_dir, 'fwd_runtime.json'))
    g.dump_bwd_runtime(os.path.join(output_dir, 'bwd_runtime.json'))
    g.dump_runtime(os.path.join(output_dir, 'global_runtime.json'))


def run_torch_graph_test(
    args
) -> None:
    """
    Runs the TorchGraph test for graph profiling.
    """
    if args.model_source == 'huggingface':  #  Hugging Face support
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:
        model = getattr(utils.transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    g = TorchGraph(model, example_input, optimizer, args.model)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
    logging.info("torch_graph: forward graph completed...")

    g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
    logging.info("torch_graph: backward graph completed...")

    g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
    logging.info("torch_graph: global graph completed...")



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    filtered_args = filter_args(args)
    
    # Add hardware information to args if using PyTorch
    if args.framework == 'PyTorch' and torch.cuda.is_available():
        setattr(args, '_cuda_available', True)
        setattr(args, '_gpu_name', torch.cuda.get_device_name(0))
        setattr(args, '_gpu_memory', f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    else:
        setattr(args, '_cuda_available', False)
    
    # Get the appropriate config display instance and display configuration
    config_display = get_config_display(args)
    config_display.display()

    if args.framework == 'PyTorch':
        if args.mode == 'runtime_profiling':
            run_torch_database_test(filtered_args)
        elif args.mode == 'graph_profiling':
            run_torch_graph_test(filtered_args)