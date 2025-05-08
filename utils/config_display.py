"""
Configuration display module for Echo Workload Tracer.

This module provides classes for displaying configuration parameters
in a professional and visually appealing format across different frameworks.
"""
from utils.common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    PYTORCH_OPS_PROFILING, PYTORCH_GRAPH_PROFILING, PYTORCH_ONLY_COMPUTE_WORKLOAD, PARALLEL_SETTING_DDP,
    Any, Dict, Optional, torch, os, json
)
from datetime import datetime


# Color and styling constants
class Colors:
    """Terminal color codes for enhanced console output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class BaseConfigDisplay:
    """
    Base class for configuration display.
    
    This class provides common functionality for displaying configuration
    parameters in a professional and standardized format.
    """
    
    def __init__(self, args: Any):
        """
        Initialize with command-line arguments.
        
        Args:
            args: The parsed command-line arguments.
        """
        self.args = args
        self.c = Colors()
    
    def display(self) -> None:
        """
        Display the configuration in a professional format.
        This method should be implemented by subclasses.
        """
        self._display_header()
        self._display_common_info()
        self._display_framework_specific()
        self._display_hardware_info()
        self._display_footer()
    
    def _display_header(self) -> None:
        """Display the header for the configuration output."""
        print(f"\n{self.c.BOLD}{self.c.UNDERLINE}Echo Workload Tracer - Configuration{self.c.END}\n")
    
    def _display_common_info(self) -> None:
        """Display common configuration information across all frameworks."""
        # Framework information
        print(f"{self.c.BOLD}{self.c.BLUE}Framework Settings:{self.c.END}")
        print(f"  • Framework:    {self.c.GREEN}{self.args.framework}{self.c.END}")
        
        # Output configuration
        print(f"\n{self.c.BOLD}{self.c.BLUE}Output Configuration:{self.c.END}")
        print(f"  • Base Path:  {self.c.GREEN}{self.args.base_path}{self.c.END}")
    
    def _display_framework_specific(self) -> None:
        """
        Display framework-specific configuration information.
        This method should be implemented by subclasses.
        """
        pass
    
    def _display_hardware_info(self) -> None:
        """Display hardware information if available."""
        # This method can be overridden by subclasses if needed
        pass
    
    def _display_footer(self) -> None:
        """Display the footer for the configuration output."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{self.c.BOLD}{self.c.YELLOW}Starting workload tracing at {timestamp} ...{self.c.END}\n")

    @staticmethod
    def format_section(items: Dict[str, Any], colors: Optional[Dict[str, str]] = None) -> None:
        """
        Format and print a section of configuration items.
        
        Args:
            items: Dictionary of item names and their values.
            colors: Optional dictionary mapping item names to color codes.
        """
        if colors is None:
            colors = {}
            
        c = Colors()
        for key, value in items.items():
            color = colors.get(key, c.GREEN)
            print(f"  • {key.ljust(13)} {color}{value}{c.END}")

    def save_config_to_json(self, args: Any) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            args: The parsed command-line arguments.
        """
        # Create config dictionary
        config = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "framework_settings": {
                "framework": args.framework,
                "base_path": args.base_path,
                "num_gpus": args.num_gpus
            }
        }
        
        # Add framework specific settings
        if args.framework == FRAME_NAME_PYTORCH:
            config["pytorch_settings"] = {
                "ddp": getattr(args, 'pytorch_ddp', False),
                "bucket_cap_mb": getattr(args, 'bucket_cap_mb', None),
                "compute_only": getattr(args, PYTORCH_ONLY_COMPUTE_WORKLOAD, False),
                "ops_profiling": getattr(args, PYTORCH_OPS_PROFILING, False),
                "graph_profiling": getattr(args, PYTORCH_GRAPH_PROFILING, False),
                "ops_profiling_path": getattr(args, 'ops_profiling_output_path', None),
                "graph_profiling_path": getattr(args, 'graph_profiling_output_path', None),
                "output_log_path": getattr(args, 'output_log_path', None)
            }
            config["model_settings"] = {
                "model": args.model,
                "model_source": args.model_source,
                "batch_size": args.batch_size,
                "sequence_length": getattr(args, 'sequence_length', 512),
                "num_repeats": getattr(args, 'num_repeats', None)
            }
            config["hardware_settings"] = {
                "cuda_available": getattr(args, '_cuda_available', False),
                "gpu_name": getattr(args, '_gpu_name', 'Unknown'),
                "gpu_memory": getattr(args, '_gpu_memory', 'Unknown')
            }
        
        suffix = ""
        if args.pytorch_ddp:
            suffix += f"_{PARALLEL_SETTING_DDP}"
        safe_model_name = args.model.replace('/', '_')
        
        config_filename = f"config_{safe_model_name}_bs{args.batch_size}_seq{args.sequence_length}{suffix}.json"
        output_dir = getattr(args, 'output_log_path', 'output/logs')
        
        # Save to file
        config_path = os.path.join(output_dir, config_filename)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


class PyTorchConfigDisplay(BaseConfigDisplay):
    """Configuration display for PyTorch framework."""
    
    def _display_framework_specific(self) -> None:
        """Display PyTorch-specific configuration."""
        c = self.c
        
        # Framework Settings Detail
        print(f"\n{c.BOLD}{c.BLUE}PyTorch Settings:{c.END}")
        
        # Display DDP and compute workload settings
        ddp_status = "Enabled" if getattr(self.args, 'pytorch_ddp', False) else "Disabled"
        print(f"  • DDP:          {c.GREEN}{ddp_status}{c.END}")
        
        if getattr(self.args, 'pytorch_ddp', False):
            bucket_cap_mb = getattr(self.args, 'bucket_cap_mb', None)
            print(f"  • Bucket Cap:   {c.GREEN}{bucket_cap_mb} MB{c.END}")
        
        compute_only = "Yes" if getattr(self.args, PYTORCH_ONLY_COMPUTE_WORKLOAD, False) else "No"
        print(f"  • Compute Only: {c.GREEN}{compute_only}{c.END}")
        
        # Profiling Mode information
        profiling_modes = []
        if getattr(self.args, 'pytorch_ops_profiling', False):
            profiling_modes.append(f"{c.GREEN}Ops Profiling{c.END}")
            ops_path = getattr(self.args, 'ops_profiling_output_path', 'Not specified')
            print(f"  • Ops Path:     {c.GREEN}{ops_path}{c.END}")
            
        if getattr(self.args, 'pytorch_graph_profiling', False):
            profiling_modes.append(f"{c.GREEN}Graph Profiling{c.END}")
            graph_path = getattr(self.args, 'graph_profiling_output_path', 'Not specified')
            print(f"  • Graph Path:   {c.GREEN}{graph_path}{c.END}")
            
        if profiling_modes:
            print(f"  • Profiling Mode:    {', '.join(profiling_modes)}")
        else:
            print(f"  • Profiling Mode:    {c.YELLOW}None{c.END}")

        output_log_path = getattr(self.args, 'output_log_path', 'Not specified')
        print(f"  • Log Path:     {c.GREEN}{output_log_path}{c.END}")
        
        # Model information
        print(f"\n{c.BOLD}{c.BLUE}Model Configuration:{c.END}")
        print(f"  • Model:        {c.GREEN}{self.args.model}{c.END}")
        print(f"  • Model Source: {c.GREEN}{self.args.model_source}{c.END}")
        print(f"  • Batch Size:   {c.GREEN}{self.args.batch_size}{c.END}")
        print(f"  • Seq Length:   {c.GREEN}{getattr(self.args, 'sequence_length', 512)}{c.END}")
        
        # Performance settings
        if hasattr(self.args, 'num_repeats'):
            print(f"\n{c.BOLD}{c.BLUE}Performance Profiling settings:{c.END}")
            print(f"  • Num Repeats:  {c.GREEN}{self.args.num_repeats}{c.END}")
            
        # GPU settings
        self.args.local_cuda_available = getattr(self.args, '_cuda_available', False)
        if self.args.local_cuda_available:
            # These values should be passed from the main script
            self.args.gpu_name = getattr(self.args, '_gpu_name', 'Unknown')
            self.args.gpu_memory = getattr(self.args, '_gpu_memory', 'Unknown')

        print(f"\n{c.BOLD}{c.BLUE}GPU Configuration in Training:{c.END}")
        print(f"  • Num GPUs:     {c.GREEN}{self.args.num_gpus}{c.END}")
        print(f"  • GPU Type:     {c.GREEN}{self.args.gpu_name}{c.END}")
        print(f"  • GPU Memory:   {c.GREEN}{self.args.gpu_memory} GB{c.END}")

    def _display_hardware_info(self) -> None:
        """Display hardware information related to PyTorch."""
        c = self.c
        
        print(f"\n{c.BOLD}{c.BLUE}Local Hardware Information:{c.END}")
        
        # We'll check torch availability when this class is instantiated
        # This avoids importing torch here
        # cuda_available = getattr(self.args, '_cuda_available', False)
        
        if self.args.local_cuda_available:
            # These values should be passed from the main script

            print(f"  • CUDA:         {c.GREEN}Available{c.END}")
            print(f"  • GPU:          {c.GREEN}{self.args.gpu_name}{c.END}")
            print(f"  • GPU Memory:   {c.GREEN}{self.args.gpu_memory} GB{c.END}")
        else:
            print(f"  • CUDA:         {c.YELLOW}Not Available{c.END}")


class DeepSpeedConfigDisplay(BaseConfigDisplay):
    """Configuration display for DeepSpeed framework."""
    
    def _display_framework_specific(self) -> None:
        """Display DeepSpeed-specific configuration."""
        c = self.c
        
        # DeepSpeed specific settings
        print(f"\n{c.BOLD}{c.BLUE}DeepSpeed Configuration:{c.END}")
        
        # Filter and display DeepSpeed-specific arguments
        deepspeed_args = {k: v for k, v in vars(self.args).items() 
                          if k.startswith('deepspeed_')}
        
        if deepspeed_args:
            for key, value in deepspeed_args.items():
                # Remove 'deepspeed_' prefix for cleaner display
                display_key = key.replace('deepspeed_', '')
                print(f"  • {display_key.ljust(15)} {c.GREEN}{value}{c.END}")
        else:
            print(f"  • {c.YELLOW}No DeepSpeed-specific settings found{c.END}")


class MegatronConfigDisplay(BaseConfigDisplay):
    """Configuration display for Megatron-LM framework."""
    
    def _display_framework_specific(self) -> None:
        """Display Megatron-LM-specific configuration."""
        c = self.c
        
        # Megatron-LM specific settings
        print(f"\n{c.BOLD}{c.BLUE}Megatron-LM Configuration:{c.END}")
        
        # Filter and display Megatron-specific arguments
        megatron_args = {k: v for k, v in vars(self.args).items() 
                         if k.startswith('megatron_')}
        
        if megatron_args:
            for key, value in megatron_args.items():
                # Remove 'megatron_' prefix for cleaner display
                display_key = key.replace('megatron_', '')
                print(f"  • {display_key.ljust(15)} {c.GREEN}{value}{c.END}")
        else:
            print(f"  • {c.YELLOW}No Megatron-LM-specific settings found{c.END}")


def get_config_display(args: Any) -> BaseConfigDisplay:
    """
    Factory function to get the appropriate configuration display class.
    
    Args:
        args: The parsed command-line arguments.
        
    Returns:
        An instance of the appropriate config display class.
    """
    framework = args.framework
    
    if framework == FRAME_NAME_PYTORCH:
        return PyTorchConfigDisplay(args)
    elif framework == FRAME_NAME_DEEPSPEED:
        return DeepSpeedConfigDisplay(args)
    elif framework == FRAME_NAME_MEGATRON:
        return MegatronConfigDisplay(args)
    else:
        # Default to base class if framework not recognized
        return BaseConfigDisplay(args)
    

def display_config(args: Any) -> None:
    """
    Display the configuration of the workload tracer.
    """
    # Add hardware information to args if using PyTorch
    if torch.cuda.is_available():
        setattr(args, '_cuda_available', True)
        setattr(args, '_gpu_name', torch.cuda.get_device_name(0))
        setattr(args, '_gpu_memory', f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    else:
        setattr(args, '_cuda_available', False)
        raise ValueError("Echo needs at least one GPU to run.")

    # Get the appropriate config display instance and display configuration
    config_display = get_config_display(args)
    config_display.display()

    # Save configuration to JSON file
    config_display.save_config_to_json(args)