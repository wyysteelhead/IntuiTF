import os
import re
import numpy as np

def get_numpy_dtype(data_type_str):
    """Convert string data type to numpy data type"""
    dtype_mapping = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'int8': np.int8,
        'int16': np.int16,
        'int32': np.int32,
        'float32': np.float32,
        'float64': np.float64,
        'ushort': np.uint16,
        'short': np.int16,
        'float': np.float32,
        'double': np.float64,
    }
    return dtype_mapping.get(data_type_str.lower(), np.float32)

def extract_dimensions(filename):
    """Extract dimension and data type information from filename"""
    patterns = [
        r"(\d+)x(\d+)x(\d+)_(\w+)",      # 256x256x124_uint8
        r"(\d+)x(\d+)x(\d+)_(\w+)\.raw", # Complete format
        r"(\d+)x(\d+)x(\d+)",            # Dimensions only
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            z = int(match.group(3))
            data_type = match.group(4) if len(match.groups()) > 3 else None
            return x, y, z, data_type
    return None

def load_and_process_volume(filepath, dims=None, dtype=None, threshold_ratio=0.1, 
                           normalize_range=(0.2, 1.0)):
    """
    Load and preprocess RAW volume data, returns only data, dims, spacing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)
    spacing = (1, 1, 1)
    
    # Try to extract information from filename
    if dims is None or dtype is None:
        extracted = extract_dimensions(filename)
        if extracted:
            x, y, z, data_type_str = extracted
            if dims is None:
                dims = (x, y, z)
            if dtype is None and data_type_str:
                dtype = get_numpy_dtype(data_type_str)
    
    if dims is None:
        raise ValueError("Dimension information must be provided (via filename or manual specification)")
    if dtype is None:
        dtype = np.float32
    
    expected_elements = dims[0] * dims[1] * dims[2]
    bytes_per_element = np.dtype(dtype).itemsize
    expected_size = expected_elements * bytes_per_element
    
    if file_size != expected_size:
        possible_dtypes = [np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64]
        for test_dtype in possible_dtypes:
            test_size = expected_elements * np.dtype(test_dtype).itemsize
            if file_size == test_size:
                dtype = test_dtype
                break
    
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    
    if len(data) > expected_elements:
        data = data[:expected_elements]
    
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=0.0)
    if np.isinf(data).any():
        data = np.nan_to_num(data, posinf=data.max(), neginf=data.min())
    
    data = data.astype(np.float32)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_range = data_max - data_min
    
    if data_range > 0:
        if data_max > 100:
            threshold = data_min + data_range * threshold_ratio
            data_processed = np.where(data < threshold, 0, data)
            non_zero_data = data_processed[data_processed > 0]
            if len(non_zero_data) > 0:
                new_min = np.min(non_zero_data)
                new_max = np.max(non_zero_data)
                range_span = normalize_range[1] - normalize_range[0]
                data = np.where(
                    data_processed == 0, 0,
                    normalize_range[0] + range_span * (data_processed - new_min) / (new_max - new_min)
                )
            else:
                data = data_processed / data_max
        elif data_max < 0.1:
            scale_factor = 1.0 / data_max
            data = data * scale_factor
        else:
            data = (data - data_min) / data_range
    
    x, y, z = dims
    try:
        reshaped_data = data.reshape(z, y, x)
    except ValueError:
        reshaped_data = data.reshape(x, y, z)
    
    return reshaped_data, dims, spacing

# # Predefined known volume configurations
# DEFAULT_KNOWN_VOLUMES = {
#     "pancreas_240x512x512_int16.raw": {
#         "dims": (240, 512, 512),
#         "dtype": "int16",
#         "spacing": (1.16, 1, 1)
#     },
#     "carp_256x256x512_uint16.raw": {
#         "dims": (256, 256, 512),
#         "dtype": "uint16", 
#         "spacing": (0.78125, 0.390625, 1)
#     },
#     "CLOUDf22_500x500x100_float32.raw": {
#         "dims": (500, 500, 100),
#         "dtype": "float32",
#         "spacing": (1, 1, 1)
#     },
#     "engine_256x256x128_uint8.raw": {
#         "dims": (256, 256, 128),
#         "dtype": "uint8",
#         "spacing": (1, 1, 1)
#     },
# }

# def load_volume_simple(filepath, **kwargs):
#     """
#     Simplified volume loading function using default configuration
    
#     Args:
#         filepath: RAW file path
#         **kwargs: Other parameters passed to load_and_process_volume
    
#     Returns:
#         dict: Processing results
#     """
#     return load_and_process_volume(filepath, known_volumes=DEFAULT_KNOWN_VOLUMES, **kwargs)