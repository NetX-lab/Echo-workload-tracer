import json
import torch
import torchvision
from torch_analysis.torch_database import TorchDatabase
from torch.autograd import Variable
from torch_analysis.timer import Timer
import torch.optim as optim
import time
import os
import argparse
from torchvision import models

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--path', type=str, default='../output/pytorch/workload_runtime',
                    help='path')
parser.add_argument("--batchsize", default=16, type=int)
parser.add_argument("--num_repeats", default=1, type=int)
args = parser.parse_args()





# transformer库包model引用example
import transformer
args.model="gpt2"
timer = Timer(args.num_repeats, args.model)

module = getattr(transformer, args.model)().cuda()
example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)
g = TorchDatabase(module, example, 'alexnet', timer, optimizer)

output_dir = os.path.join(args.path, g.name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

g.dump_fwd_runtime(os.path.join(output_dir, 'fwd_runtime.json'))
print("torch_graph: 已完成mytest_fwd_graph...")

g.dump_bwd_runtime(os.path.join(output_dir, 'bwd_runtime.json'))
print("torch_graph: 已完成mytest_bwd_graph...")

g.dump_runtime(os.path.join(output_dir, 'global_runtime.json'))
print("torch_graph: 已完成mytest_fbwd_graph...")
