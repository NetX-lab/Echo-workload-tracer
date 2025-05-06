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
from utils import get_logger

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
        output_path: Dict[str, str],
        parallel_setting: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the base tracer.
        
        Args:
            model: The model to be traced
            model_name: Name of the model
            output_path: Dictionary mapping output types to directories to save results
            parallel_setting: Parallel training setting (e.g., 'ddp')
            **kwargs: Additional framework-specific arguments
        """
        self.model = model
        self.model_name = model_name
        self.output_path = output_path
        self.parallel_setting = parallel_setting
        self.start_time = datetime.now()
        self.trace_data = {
            "metadata": {
                "model_name": model_name,
                "start_time": self.start_time.isoformat(),
                "framework": self._get_framework_name(),
                "parallel_setting": parallel_setting
            },
            "events": []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Log initialization
        self.logger.info(f"Initialized {self._get_framework_name()} tracer for model: {model_name}")
    
    def _setup_logging(self) -> None:
        """Setup logging for the tracer."""
        self.logger = logging.getLogger(f"{self._get_framework_name()} tracer")
        self.logger = get_logger(
            f"{self._get_framework_name()} tracer", 
            log_dir=list(self.output_path.values())[0] if self.output_path else "."
        )
    
    @abstractmethod
    def _get_framework_name(self) -> str:
        """
        Return the name of the framework this tracer is for.
        
        Returns:
            str: Framework name (e.g., 'PyTorch', 'DeepSpeed', 'Megatron-LM')
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
        
    def display_events(self) -> None:
        """
        Display all events that have been recorded during tracing.
        
        This function logs all the events with their timestamps and data
        for debugging and monitoring purposes.
        """
        # self.logger.info(f"Displaying {len(self.trace_data['events'])} events for {self.model_name}:")
        
        for idx, event in enumerate(self.trace_data["events"]):
            event_time = datetime.fromtimestamp(event["timestamp"]).strftime('%Y-%m-%d %H:%M:%S.%f')
            event_type = event["type"]
            event_data_str = json.dumps(event["data"], indent=None)
            
            self.logger.info(f"Event #{idx+1}: [{event_time}] {event_type} - {event_data_str}")
        
        # self.logger.info(f"End of events for {self.model_name}")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the tracer to perform the tracing operation.
        
        Returns:
            Dict[str, Any]: The result of the tracing operation
        """
        pass 