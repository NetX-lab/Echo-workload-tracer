#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common imports and constants used throughout the Echo Workload Tracer project.
"""

# Standard library imports
import os
import sys
import json
import logging
import warnings
import subprocess
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

# Third-party imports
try:
    import torch
    import torch.optim as optim
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install the required dependencies.")
    sys.exit(1)

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)



# Framework constants
FRAME_NAME_PYTORCH = 'PyTorch'
FRAME_NAME_DEEPSPEED = 'DeepSpeed'
FRAME_NAME_MEGATRON = 'Megatron-LM'

# Profiling modes
PYTORCH_OPS_PROFILING = 'pytorch_ops_profiling'
PYTORCH_GRAPH_PROFILING = 'pytorch_graph_profiling'
PYTORCH_ONLY_COMPUTE_WORKLOAD = 'pytorch_only_compute_workload'


# Model sources
MODEL_SOURCE_HUGGINGFACE = 'huggingface'
MODEL_SOURCE_LOCAL = 'local'

# File constants
DEFAULT_OUTPUT_DIR = 'output'
CONFIG_FILE_PATH = 'configs'

# Parallel settings
PARALLEL_SETTING_DDP = 'pytorch_ddp'


# Common utility functions
def ensure_dir_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        # logger.info(f"Created directory: {directory}")