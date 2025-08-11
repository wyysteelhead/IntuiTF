import os
import re
import numpy as np

def get_numpy_dtype(data_type_str):
    """将字符串数据类型转换为numpy数据类型"""
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
    """从文件名提取维度和数据类型信息"""
    patterns = [
        r"(\d+)x(\d+)x(\d+)_(\w+)",      # 256x256x124_uint8
        r"(\d+)x(\d+)x(\d+)_(\w+)\.raw", # 完整格式
        r"(\d+)x(\d+)x(\d+)",            # 只有维度
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
    加载并预处理RAW体积数据，只返回data, dims, spacing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)
    spacing = (1, 1, 1)
    
    # 尝试从文件名提取信息
    if dims is None or dtype is None:
        extracted = extract_dimensions(filename)
        if extracted:
            x, y, z, data_type_str = extracted
            if dims is None:
                dims = (x, y, z)
            if dtype is None and data_type_str:
                dtype = get_numpy_dtype(data_type_str)
    
    if dims is None:
        raise ValueError("必须提供维度信息（通过文件名或手动指定）")
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

# # 预定义的已知体积配置
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
#     简化的体积加载函数，使用默认配置
    
#     Args:
#         filepath: RAW文件路径
#         **kwargs: 传递给load_and_process_volume的其他参数
    
#     Returns:
#         dict: 处理结果
#     """
#     return load_and_process_volume(filepath, known_volumes=DEFAULT_KNOWN_VOLUMES, **kwargs)