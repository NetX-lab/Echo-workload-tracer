#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import logging
from typing import Optional
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .torch_database import TorchDatabase
from .profiling_timer import Timer
import os
import json

class DDPTorchDatabase(TorchDatabase):
    """
    Generate a torch database for Distributed Data Parallel (DDP) model with torch.fx.Interpreter
    Extends the TorchDatabase class to provide profiling for DDP models.
    
    Basic usage:
        module = DDP(torchvision.models.resnet50(pretrained=True).cuda())
        example = torch.rand(1, 3, 224, 224).cuda()
        optimizer = optim.SGD(module.parameters(), lr=0.01)
        
        timer = Timer(100, 'ddp_resnet50')
        
        g = DDPTorchDatabase(module, example, 'ddp_resnet50', timer, optimizer, logger)
    """
    def __init__(
        self, 
        module: torch.nn.Module, 
        example: torch.Tensor, 
        name: str, 
        timer: Timer, 
        optimizer: Optimizer,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DDPTorchDatabase.
        
        Args:
            module: The DDP wrapped module to profile
            example: Input tensor example used for profiling
            name: Name identifier for this profiling session
            timer: Timer object for performance measurement
            optimizer: Optimizer object used for the module
            logger: Logger object for logging progress (optional)
        """
        # Store the DDP module reference before passing to parent
        self._ddp_module = module if isinstance(module, DDP) else None
        
        # Call parent class constructor with all arguments
        super().__init__(module, example, name, timer, optimizer, logger)
        
        if self._ddp_module:
            self.logger.info(f"Initialized DDPTorchDatabase for DDP module: {name}")
        
    def get_ddp_module(self):
        """
        Returns the underlying module wrapped by DDP.
        """
        if hasattr(self.module, 'module'):
            return self.module.module
        return self.module
    
    def _get_bp_node_time(self):
        """
        Overrides parent method to handle DDP specific backward pass profiling.
        """
        self.logger.info("Profiling DDP backward pass...")
        super()._get_bp_node_time()
        
    def _get_optimizer_node_time(self):
        """
        Overrides parent method to handle DDP specific optimizer profiling.
        """
        self.logger.info("Profiling DDP optimizer operations...")
        super()._get_optimizer_node_time()

    def dump_ddp_runtime(self, path: str) -> None:
        """
        Dump DDP-specific runtime metrics to a JSON file.
        
        Args:
            path: Path to the output JSON file
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Filter out DDP specific operations
        ddp_ops = {}
        for op_name, runtime in self._get_overall_database().items():
            if op_name.startswith("ddp_") or "AllReduce" in op_name:
                ddp_ops[op_name] = runtime
        
        with open(path, 'w') as file:
            json.dump(ddp_ops, file, indent=4)