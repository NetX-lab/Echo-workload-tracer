#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracer Core Package - The core implementation of the workload tracer.
"""

from tracer_core.base import BaseTracer
from tracer_core.pytorch_tracer import PyTorchTracer

# Conditionally import if available
try:
    from tracer_core.deepspeed_tracer import DeepSpeedTracer
except ImportError:
    pass

try:
    from tracer_core.megatron_tracer import MegatronTracer
except ImportError:
    pass

from tracer_core.tracer_initializer import TracerInitializer, create_tracer

__all__ = [
    "BaseTracer",
    "PyTorchTracer",
    "TracerInitializer",
    "create_tracer",
    "torch_analysis"
] 