"""
Configuration display module for Echo Workload Tracer.

This module provides classes for displaying configuration parameters
in a professional and visually appealing format across different frameworks.
"""
from common import (
    FRAME_NAME_PYTORCH, FRAME_NAME_DEEPSPEED, FRAME_NAME_MEGATRON,
    MODE_RUNTIME_PROFILING, MODE_GRAPH_PROFILING,
    MODEL_SOURCE_HUGGINGFACE, MODEL_SOURCE_LOCAL,
    Any, Dict, Optional
)


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
        print(f"  • Output Path:  {self.c.GREEN}{self.args.path}{self.c.END}")
    
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
        print(f"\n{self.c.BOLD}{self.c.YELLOW}Starting workload tracing...{self.c.END}\n")

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


class PyTorchConfigDisplay(BaseConfigDisplay):
    """Configuration display for PyTorch framework."""
    
    def _display_framework_specific(self) -> None:
        """Display PyTorch-specific configuration."""
        c = self.c
        
        # Mode information
        print(f"  • Mode:         {c.GREEN}{self.args.mode}{c.END}")
        
        # Model information
        print(f"\n{c.BOLD}{c.BLUE}Model Configuration:{c.END}")
        print(f"  • Model:        {c.GREEN}{self.args.model}{c.END}")
        print(f"  • Model Source: {c.GREEN}{self.args.model_source}{c.END}")
        print(f"  • Batch Size:   {c.GREEN}{self.args.batchsize}{c.END}")
        
        # Performance settings
        if hasattr(self.args, 'num_repeats'):
            print(f"\n{c.BOLD}{c.BLUE}Performance Settings:{c.END}")
            print(f"  • Num Repeats:  {c.GREEN}{self.args.num_repeats}{c.END}")
    
    def _display_hardware_info(self) -> None:
        """Display hardware information related to PyTorch."""
        c = self.c
        
        print(f"\n{c.BOLD}{c.BLUE}Hardware Information:{c.END}")
        
        # We'll check torch availability when this class is instantiated
        # This avoids importing torch here
        cuda_available = getattr(self.args, '_cuda_available', False)
        
        if cuda_available:
            # These values should be passed from the main script
            gpu_name = getattr(self.args, '_gpu_name', 'Unknown')
            gpu_memory = getattr(self.args, '_gpu_memory', 'Unknown')
            
            print(f"  • CUDA:         {c.GREEN}Available{c.END}")
            print(f"  • GPU:          {c.GREEN}{gpu_name}{c.END}")
            print(f"  • GPU Memory:   {c.GREEN}{gpu_memory} GB{c.END}")
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