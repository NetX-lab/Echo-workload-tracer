#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.fx
from torch.fx.node import Node, map_aggregate
from torch.fx import symbolic_trace
from shape_prop import ShapeProp, TensorMetadata, extract_tensor_metadata
from typename import typename
import Node
from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
from .torch_graph import TorchGraph

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from typing import Optional
import os
import json


def get_bucket_indices(model):
    """
    Extract bucket indices from a DDP model.
    
    Args:
        model: A DistributedDataParallel model
        
    Returns:
        parameters: List of parameters that require gradients
        bucket_indices: List of parameter indices in each bucket (reversed)
    """
    if not isinstance(model, DDP):
        raise TypeError("Input model must be DistributedDataParallel model")

    # Build tuple of (module, parameter) for all parameters that require grads
    modules_and_parameters = [
        [
            (module, parameter)
            for module_name, module in model.module.named_modules()
            for parameter in [
                param
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in model.parameters_to_ignore
            ]
        ]
    ]
    
    # Deduplicate any parameters that might be shared across child modules
    memo = set()
    modules_and_parameters = [
        [(m, p) for m, p in replica_mps if p not in memo and not memo.add(p)]
        for replica_mps in modules_and_parameters
    ]

    # Build list of parameters
    parameters = [
        list(parameter for _, parameter in replica)
        for replica in modules_and_parameters
    ]

    def produces_sparse_gradient(module):
        """Check if a module will produce a sparse gradient."""
        if isinstance(module, torch.nn.Embedding) or isinstance(
            module, torch.nn.EmbeddingBag
        ):
            return module.sparse
        return False

    # Build list of booleans indicating whether to expect sparse gradients
    expect_sparse_gradient = [
        list(produces_sparse_gradient(module) for module, _ in replica)
        for replica in modules_and_parameters
    ]

    bucket_indices, _ = dist._compute_bucket_assignment_by_size(
            parameters[0],
            [dist._DEFAULT_FIRST_BUCKET_BYTES, model.bucket_bytes_cap],
            expect_sparse_gradient[0],
        )

    return parameters[0], list(reversed(bucket_indices))


class DDPGraph(TorchGraph):
    """
    DDPGraph extends TorchGraph to provide distributed data parallel (DDP) bucket-aware graph tracing.

    This class traces a PyTorch model wrapped with DistributedDataParallel (DDP), capturing not only the 
    standard forward and backward computation graphs, but also the communication buckets and AllReduce 
    operations used in DDP for gradient synchronization.

    Key Features:
        - Automatically initializes the distributed process group and wraps the model with DDP.
        - Extracts DDP bucket information, including parameter-to-bucket mapping and bucket sizes.
        - Augments the computation graph with DDP-specific nodes representing communication steps 
          (e.g., pre-bucket and AllReduce nodes).
        - Maintains compatibility with all TorchGraph features, such as graph export and optimizer tracing.
        - Provides methods to export DDP-specific graph components for further analysis or visualization.

    Args:
        module (torch.nn.Module): The PyTorch model to be traced.
        example (torch.Tensor): Example input tensor for tracing.
        optimizer (torch.optim.Optimizer): Optimizer associated with the model.
        name (str): Name identifier for the model/graph.
        local_rank (int): Local rank of the current process (for DDP).
        bucket_cap_mb (int): DDP bucket size in MB for gradient communication.
        logger (Optional[logging.Logger]): Optional logger for progress and debug information.
    """

    def __init__(
        self, 
        module: torch.nn.Module, 
        example: torch.tensor, 
        optimizer: torch.optim, 
        name: str, 
        local_rank: int,
        bucket_cap_mb: int = 25,
        logger: Optional[logging.Logger] = None
    ):
        # env setting for using DDP
        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29555'
        
        # Initialize distributed process group
        dist.init_process_group(backend='nccl')
        
        # Wrap module with DDP with specified bucket_cap_mb
        ddp_module = DDP(module, device_ids=[local_rank], output_device=local_rank, bucket_cap_mb=bucket_cap_mb)
        
        # Get bucket information for DDP
        self._parameters, self._bucket_indices = get_bucket_indices(ddp_module)
        
        # Create parameter ID to index mapping
        self._param_map = {id(param): index for index, param in enumerate(self._parameters)}
        
        self._parameter_to_bucket = [0 for _ in self._parameters]
        self._bucket_size = [0 for _ in self._bucket_indices]
        
        # Build parameter to bucket mapping and calculate bucket sizes
        for bucket_index, bucket_indice in enumerate(self._bucket_indices):
            for i in bucket_indice:
                self._parameter_to_bucket[i] = bucket_index
                self._bucket_size[bucket_index] += self._parameters[i].nelement()
        
        self._bucket_list = [[] for _ in self._bucket_indices]
        
        # Initialize the parent class with unwrapped module to use its functionality
        super().__init__(module, example, optimizer, name, logger)
        
        # Create DDP-specific graph
        self.logger.info("Starting comm. bucket ddp nodes graph creation...")
        self._create_DDP_graph()
        self.logger.info("comm. bucket ddp nodes graph creation completed")

    def _make_backward_hook(self, node):
        """Create a hook function for tracking gradient operations."""
        def hook(inputs, outputs):
            if self._get_bp_node_op(node) not in self._backward_op_dict:
                self._backward_op_dict[self._get_bp_node_op(node)] = 0
            else:
                self._backward_op_dict[self._get_bp_node_op(node)] += 1
            self._backward_graph_dict[node]['name'] = \
                self._get_bp_node_op(node) + str(self._backward_op_dict[self._get_bp_node_op(node)])
            self._backward_graph_dict[node]['input_meta'] = \
                self._get_tensor_meta(outputs)
            self._backward_graph_dict[node]['output_meta'] = \
                self._get_tensor_meta(inputs)

            self._grad_fn_list.append(node)

            # Track AccumulateGrad operations in backward process
            if hasattr(node, 'variable'):
                if id(node.variable) in self._param_map:
                    index_ = self._param_map[id(node.variable)]
                    self._bucket_list[self._parameter_to_bucket[index_]].append(self._get_bp_node_name(node))
                else:
                    raise RuntimeError("Found unseen parameters in backward graph")

        return hook

    def _create_DDP_graph(self):
        """Create DDP-specific nodes in the graph representing AllReduce operations."""
        for index_, bucket in enumerate(self._bucket_list):
            pre_bucket_node = self._NodeEngineer.construct_node(
                name="ddp_pre_" + str(index_),
                op="ddp_pre",
                input_nodes=bucket,
                output_nodes=["ddp_Allreduce_" + str(index_)],
                input_types=[],
                input_shapes=[],
                output_types=[],
                output_shapes=[],
                attrs={'bucket_size': self._bucket_size[index_]}
            )
            
            bucket_node = self._NodeEngineer.construct_node(
                name="ddp_Allreduce_" + str(index_),
                op="ddp_Allreduce",
                input_nodes=["ddp_pre_" + str(index_)],
                output_nodes=[],
                input_types=[],
                input_shapes=[],
                output_types=[],
                output_shapes=[],
                attrs={'bucket_size': self._bucket_size[index_]}
            )

            # Connect AccumulateGrad ops to ddp_pre node
            for accumulate_grad in bucket:
                self._graph_dict[accumulate_grad].output_nodes.append(pre_bucket_node.name)

            # Add DDP nodes to graph
            self._graph_dict[pre_bucket_node.name] = pre_bucket_node
            self._graph_dict[bucket_node.name] = bucket_node

            # Connect DDP nodes sequentially
            if index_ != 0:
                self._graph_dict["ddp_pre_" + str(index_)].input_nodes.append("ddp_Allreduce_" + str(index_-1))
                self._graph_dict["ddp_Allreduce_" + str(index_-1)].output_nodes.append("ddp_pre_" + str(index_))

    def dump_ddp_graph(self, path: str) -> None:
        """
        Dump the DDP-specific graph to a JSON file.
        
        Args:
            path: Path to the output JSON file
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        ddp_nodes = {}
        # Extract all DDP-related nodes
        for name, node in self._graph_dict.items():
            if name.startswith("ddp_"):
                ddp_nodes[name] = node
        
        with open(path, 'w') as file:
            json.dump([node.to_json() for node in ddp_nodes.values()], file, indent=4)
