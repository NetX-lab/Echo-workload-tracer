#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Tracer - Abstract base class for all framework-specific tracers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import os
import logging
import json
import time
import torch
from datetime import datetime


class BaseTracer(ABC):
    """
    Abstract base class for all workload tracers.
    
    This class defines the common interface that all framework-specific
    tracers must implement. It provides basic functionality for tracing,
    logging, and output management.
    """
    
    def __init__(
        self, 
        model: Any,
        model_name: str,
        output_path: str,
        mode: str,
        **kwargs
    ) -> None:
        """
        Initialize the base tracer.
        
        Args:
            model: The model to be traced
            model_name: Name of the model
            output_path: Directory to save tracing results
            mode: Tracing mode (e.g., 'runtime_profiling', 'graph_profiling')
            **kwargs: Additional framework-specific arguments
        """
        self.model = model
        self.model_name = model_name
        self.output_path = output_path
        self.mode = mode
        self.start_time = datetime.now()
        self.trace_data = {
            "metadata": {
                "model_name": model_name,
                "tracing_mode": mode,
                "start_time": self.start_time.isoformat(),
                "framework": self._get_framework_name()
            },
            "events": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Log initialization
        self.logger.info(f"Initialized {self._get_framework_name()} tracer for model: {model_name}")
    
    def _setup_logging(self) -> None:
        """Setup logging for the tracer."""
        self.logger = logging.getLogger(f"{self._get_framework_name()}_tracer")
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler
        log_file = os.path.join(self.output_path, f"{self.model_name}_trace.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    @abstractmethod
    def _get_framework_name(self) -> str:
        """
        Return the name of the framework this tracer is for.
        
        Returns:
            str: Framework name (e.g., 'PyTorch', 'DeepSpeed', 'Megatron-LM')
        """
        pass
    
    @abstractmethod
    def setup_hooks(self) -> None:
        """
        Set up the necessary hooks for tracing.
        
        This method should implement the framework-specific logic for setting up
        hooks to capture the desired information during model execution.
        """
        pass
    
    @abstractmethod
    def trace_forward(self, *args, **kwargs) -> Any:
        """
        Trace the forward pass of the model.
        
        Args:
            *args: Arguments to pass to the model's forward method
            **kwargs: Keyword arguments to pass to the model's forward method
            
        Returns:
            Any: The output of the model's forward pass
        """
        pass
    
    @abstractmethod
    def trace_backward(self, loss: torch.Tensor) -> None:
        """
        Trace the backward pass of the model.
        
        Args:
            loss: The loss tensor to backpropagate
        """
        pass
    
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Add a tracing event to the trace data.
        
        Args:
            event_type: Type of event (e.g., 'forward_start', 'backward_end')
            event_data: Data associated with the event
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": event_data
        }
        self.trace_data["events"].append(event)
        self.logger.debug(f"Added event: {event_type}")
    
    def save_trace(self, filename: Optional[str] = None) -> None:
        """
        Save the trace data to a JSON file.
        
        Args:
            filename: Name of the file to save the trace data to. If None,
                     a default name based on the model name and timestamp will be used.
        """
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{self._get_framework_name().lower()}_{timestamp}.json"
        
        # Update end time
        self.trace_data["metadata"]["end_time"] = datetime.now().isoformat()
        
        filepath = os.path.join(self.output_path, filename)
        with open(filepath, 'w') as f:
            json.dump(self.trace_data, f, indent=2)
        
        self.logger.info(f"Saved trace data to {filepath}")
    
    def run(self, *args, **kwargs) -> Any:
        """
        Run the tracer to perform the tracing operation.
        
        This method should be overridden by subclasses to implement
        the specific tracing logic for each framework.
        
        Args:
            *args: Arguments to pass to the tracing methods
            **kwargs: Keyword arguments to pass to the tracing methods
            
        Returns:
            Any: The result of the tracing operation
        """
        self.logger.info(f"Starting trace for {self.model_name}")
        # This is a placeholder that should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement run()") 