import torch
import builtins
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set

def _find_module_of_method(
    orig_method: Callable[..., Any]
) -> str:
    """
    Find the module name of the given method.
    """
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

def _get_qualified_name(
    func: Callable[..., Any]
) -> str:
    """
    Get the fully qualified name of the given function.
    """
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
     # WAR for bug in how torch.ops assigns module
    module = module.replace('torch._ops', 'torch.ops')
    return f'{module}.{name}'

def typename(
    target: Any
) -> str:
    """
    Get the type name of the given target.
    """
    if isinstance(target, torch.nn.Module):
        return torch.typename(target)
    if isinstance(target, str):
        return target
    return _get_qualified_name(target)