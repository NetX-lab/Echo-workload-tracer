import logging
import os
import sys
from datetime import datetime

# Colors for console output (same as your existing color codes)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def setup_logger(name, log_dir=None):
    """
    Set up and return a logger with the given name.
    
    Args:
        name: Name for the logger
        log_dir: Directory for log files (if None, only console logging is configured)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a global logger for general use
main_logger = setup_logger('echo_tracer')

def get_logger(name=None, log_dir=None):
    """
    Get a logger with the given name, or the main logger if name is None.
    
    Args:
        name: Name for the logger (if None, returns main logger)
        log_dir: Directory for log files (if None, only console logging is used)
    
    Returns:
        logger: Configured logger instance
    """
    if name is None:
        return main_logger
    return setup_logger(name, log_dir)