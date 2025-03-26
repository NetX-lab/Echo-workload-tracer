import json
import torch
import torchvision
import sys
sys.path.append('/root/Echo-workload-tracer/torch_analysis')
from torch_analysis.torch_graph import TorchGraph
import torch.optim as optim
import argparse
import os
import utils.transformer
# transformer库包model引用example

def main(args):
    args.model = "gpt2"

    module = getattr(utils.transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
    g = TorchGraph(module, example, optimizer, 'gpt2')

    output_dir = os.path.join(args.path, g.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
    print("torch_graph: 已完成mytest_fwd_graph...")

    g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
    print("torch_graph: 已完成mytest_bwd_graph...")

    g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
    print("torch_graph: 已完成mytest_fbwd_graph...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
    parser.add_argument('--path', type=str, default='../output/pytorch/workload_graph',
                    help='path')
    parser.add_argument("--batchsize", default=16, type=int)
    args = parser.parse_args()
    main(args)
# import json
# import torch
# import torchvision
# from torch_graph import TorchGraph
# import torch.optim as optim
# import argparse


# parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--model', type=str, default='resnet50',
#                     help='model to benchmark')
# args = parser.parse_args()

# args.model="gpt2"
# args.type="NLP"

# from torchvision import models
# module = getattr(models, args.model)().cuda()
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
# g.dump_graph(args.model + "test.json")


# # 自定义model workload trace example
# import torch.nn as nn
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.linear = nn.Linear(10*32*32, 5)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = x.view(x.size(0), -1)
#         return self.linear(x)

# module = SimpleModel()
# example = torch.randn(1, 3, 32, 32) # .cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module=module, example=example, optimizer=optimizer, name='SimpleModel')

# # 创建输出目录
# output_dir = os.path.join(args.path, g._name)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
# print("torch_graph: 已完成mytest_fwd_graph...")

# g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
# print("torch_graph: 已完成mytest_bwd_graph...")

# g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
# print("torch_graph: 已完成mytest_fbwd_graph...")



# # CV库包model引用example
# from torchvision import models
# import transformer
# args.model="resnet50"

# module = getattr(models, args.model)().cuda()
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)
# g = TorchGraph(module, example, optimizer, 'resnet50')

# output_dir = os.path.join(args.path, g._name)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# g.dump_fwd_graph(os.path.join(output_dir, 'fwd_graph2.json'))
# print("torch_graph: 已完成mytest_fwd_graph...")

# g.dump_bwd_graph(os.path.join(output_dir, 'bwd_graph2.json'))
# print("torch_graph: 已完成mytest_bwd_graph...")

# g.dump_graph(os.path.join(output_dir, 'global_graph2.json'))
# print("torch_graph: 已完成mytest_fbwd_graph...")

