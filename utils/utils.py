import os
import torch
import torch.optim as optim
from typing import Any

from utils.common import (
    AutoModel, AutoTokenizer, ensure_dir_exists, MODEL_SOURCE_HUGGINGFACE, PYTORCH_OPS_PROFILING
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

    # Get sequence length from args or use default
    sequence_length = getattr(args, 'sequence_length', 512)

    # Load model and prepare input tensor
    if args.model_source == MODEL_SOURCE_HUGGINGFACE:
        model, tokenizer = load_huggingface_model(args.model)
        
        # Get the vocabulary size to ensure generated token IDs are valid
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 30000
        
        # Generate random input tensor with specified batch_size and sequence_length
        # This matches the method used for local models
        example_input = (torch.LongTensor(args.batch_size, sequence_length).random_() % vocab_size).cuda()
        
        print(f"Generated random input tensor with shape: {example_input.shape}")
        
    else:  # Local model
        model = getattr(transformer, args.model)().cuda()
        # For local models, directly create tensor with specified sequence length
        example_input = (torch.LongTensor(args.batch_size, sequence_length).random_() % 1000).cuda()
    
    # Create optimizer if needed
    if args.pytorch_ops_profiling or args.pytorch_graph_profiling:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None
    
    # Attach to args for easy access
    setattr(args, 'model_instance', model)
    setattr(args, 'example_input', example_input)
    setattr(args, 'optimizer', optimizer)
