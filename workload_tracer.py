#!/root/miniconda3/envs/tracer_zz/bin/python3
import warnings
warnings.filterwarnings("ignore")
import subprocess
import os
import sys
import torch
import json
import logging

sys.path.append('/root/Echo-workload-tracer/torch_analysis')
from tracer_arguments import get_parser, filter_args
from torch_analysis.torch_database import TorchDatabase
from torch_analysis.torch_graph import TorchGraph
from torch_analysis.profiling_timer import Timer
import torch.optim as optim
import utils.transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_torch_database_test(args):
    timer = Timer(args.num_repeats, args.model)

    module = getattr(utils.transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
    g = TorchDatabase(module, example, args.model, timer, optimizer)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_runtime(os.path.join(output_dir, 'fwd_runtime.json'))

    g.dump_bwd_runtime(os.path.join(output_dir, 'bwd_runtime.json'))

    g.dump_runtime(os.path.join(output_dir, 'global_runtime.json'))


def run_torch_graph_test(args):
    module = getattr(utils.transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
    g = TorchGraph(module, example, optimizer, args.model)

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
    logging.info("torch_graph: 已完成fwd_graph...")

    g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
    logging.info("torch_graph: 已完成bwd_graph...")

    g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
    logging.info("torch_graph: 已完成fbwd_graph...")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    filtered_args = filter_args(args)

    if args.framework == 'Pytorch':
        if args.mode == 'runtime_profiling':
            run_torch_database_test(filtered_args)
        elif args.mode == 'graph_profiling':
            run_torch_graph_test(filtered_args)