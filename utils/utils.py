import os
import torch
import torch.optim as optim
from typing import Any

from utils.common import (
    AutoModel, AutoTokenizer, ensure_dir_exists, MODEL_SOURCE_HUGGINGFACE, MODE_RUNTIME_PROFILING
)   
import utils.transformer as transformer


def load_huggingface_model(
    model_name: str
) -> tuple:
    """
    Load a Hugging Face model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_model_and_inputs(args: Any) -> None:
    """
    Prepare the model and example inputs based on arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(args.path)
    
    # Load model and prepare input tensor
    if args.model_source == MODEL_SOURCE_HUGGINGFACE:
        model, tokenizer = load_huggingface_model(args.model)
        example_input = tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    else:  # Local model
        model = getattr(transformer, args.model)().cuda()
        example_input = (torch.LongTensor(args.batchsize, 512).random_() % 1000).cuda()
    
    # Create optimizer if needed
    if args.mode == MODE_RUNTIME_PROFILING:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None
    
    # Attach to args for easy access
    setattr(args, 'model_instance', model)
    setattr(args, 'example_input', example_input)
    setattr(args, 'optimizer', optimizer)
