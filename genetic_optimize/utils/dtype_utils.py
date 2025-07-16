import numpy as np
import torch

def get_renderer_dtypes(renderer_type="diffdvr"):
    """
    Get the appropriate data types for the specified renderer.
    
    Args:
        renderer_type (str): The type of renderer ("diffdvr" or "anari")
        
    Returns:
        tuple: (torch_dtype, numpy_dtype)
    """
    if renderer_type == "diffdvr":
        # Import here to avoid importing when using anari
        import pyrenderer
        return (torch.float64 if pyrenderer.use_double_precision() else torch.float32,
                np.float64 if pyrenderer.use_double_precision() else np.float32)
    else:  # anari or other renderers
        return torch.float32, np.float32

# # Default dtypes that can be imported directly
# renderer_dtype_torch, renderer_dtype_np = get_renderer_dtypes() 