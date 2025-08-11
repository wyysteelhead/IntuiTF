import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 尝试导入pynari
try:
    import pynari
except ImportError:
    print("警告: 无法导入pynari，请确保已正确安装")

def spherical_to_cartesian(pitch, yaw, distance, center):
    """球坐标转笛卡尔坐标"""
    x = distance * math.cos(pitch) * math.sin(yaw)
    y = distance * math.sin(pitch)
    z = distance * math.cos(pitch) * math.cos(yaw)
    return (center[0] + x, center[1] + y, center[2] + z)

def setup_volume_renderer(volume_data, volume_dims, tf,
                         center_at_origin=True, device=None):
    """
    设置体积渲染器
    
    Args:
        volume_data: 体积数据numpy数组，形状为(depth, height, width)
        volume_dims: 原始体积维度 (width, height, depth)
        tf: 传输函数numpy数组，None为自动选择
        center_at_origin: 是否将体积中心置于原点
        device: ANARI设备，None为创建新设备
    
    Returns:
        dict: {
            'device': ANARI设备,
            'world': 世界对象,
            'volume': 体积对象,
            'spatial_field': 空间场对象,
            'volume_info': 体积信息字典
        }
    """
    if pynari is None:
        raise ImportError("pynari未安装，无法使用渲染功能")
    
    # 创建设备
    if device is None:
        device = pynari.newDevice('default')
    
    # 计算单元格大小和原点
    max_dim = max(volume_dims)
    cellSize = (2/max_dim, 2/max_dim, 2/max_dim)
    
    if center_at_origin:
        # 居中设置
        actual_x = volume_dims[0] * cellSize[0]
        actual_y = volume_dims[1] * cellSize[1]
        actual_z = volume_dims[2] * cellSize[2]
        origin = (-actual_x/2, -actual_y/2, -actual_z/2)
    else:
        origin = (-1, -1, -1)
    
    # 创建ANARI对象
    structured_data = device.newArray(pynari.float, volume_data)
    spatial_field = device.newSpatialField('structuredRegular')
    spatial_field.setParameter('origin', pynari.float3, origin)
    spatial_field.setParameter('spacing', pynari.float3, cellSize)
    spatial_field.setParameter('data', pynari.ARRAY3D, structured_data)
    spatial_field.commitParameters()

    xf_array = device.newArray(pynari.float4, tf)
    volume = device.newVolume('transferFunction1D')
    volume.setParameter('color', pynari.ARRAY, xf_array)
    volume.setParameter('value', pynari.SPATIAL_FIELD, spatial_field)
    volume.setParameter('unitDistance', pynari.FLOAT32, 10.)
    volume.commitParameters()

    world = device.newWorld()
    volume_array = device.newArray(pynari.VOLUME, [volume])
    world.setParameter('volume', pynari.ARRAY3D, volume_array)

    # 设置光照
    light = device.newLight('directional')
    light.setParameter('direction', pynari.float3, (1., -1., -1.))
    light.commitParameters()
    light_array = device.newArray(pynari.LIGHT, [light])
    world.setParameter('light', pynari.ARRAY1D, light_array)
    world.commitParameters()
    
    volume_info = {
        'dims': volume_dims,
        'cellSize': cellSize,
        'origin': origin,
        'center': (0, 0, 0) if center_at_origin else (origin[0] + actual_x/2, origin[1] + actual_y/2, origin[2] + actual_z/2)
    }
    
    return {
        'device': device,
        'world': world,
        'volume': volume,
        'spatial_field': spatial_field,
        'volume_info': volume_info
    }

def setup_camera(device, W, H, look_from, look_at, look_up, fovy=40.0):
    """
    设置相机
    
    Args:
        device: ANARI设备
        fb_size: 帧缓冲区尺寸 (width, height)
        look_from: 相机位置 (x, y, z)
        look_at: 观察目标 (x, y, z)
        look_up: 上方向 (x, y, z)
        fovy: 视野角度（度）
    
    Returns:
        camera: 相机对象
    """
    fb_size = (W, H)
    camera = device.newCamera('perspective')
    camera.setParameter('aspect', pynari.FLOAT32, fb_size[0]/fb_size[1])
    camera.setParameter('position', pynari.FLOAT32_VEC3, look_from)
    
    direction = [look_at[0] - look_from[0],
                 look_at[1] - look_from[1],
                 look_at[2] - look_from[2]]
    camera.setParameter('direction', pynari.float3, direction)
    camera.setParameter('up', pynari.float3, look_up)
    camera.setParameter('fovy', pynari.float, fovy * math.pi / 180)
    camera.commitParameters()
    
    return camera

def setup_renderer(device, camera, world, fb_size, bg_color, pixel_samples=None):
    """
    设置渲染器
    
    Args:
        device: ANARI设备
        fb_size: 帧缓冲区尺寸 (width, height)
        bg_color: 背景颜色，输入为字符串"(r,g,b)"格式
        pixel_samples: 像素采样数，None为自动选择
    
    Returns:
        dict: {'renderer': 渲染器, 'frame': 帧对象}
    """
    bg_color = bg_color.strip("() ").split(',')
    bg_color = np.array(bg_color, dtype=np.float32)
    bg_color = np.concatenate((bg_color, [1]))
    bg_color = np.clip(bg_color, 0, 1)
    bg_values = np.array((bg_color,bg_color), dtype=np.float32).reshape((2,1,4))
    bg_gradient = device.newArray(pynari.float4, bg_values)

    renderer = device.newRenderer('default')
    renderer.setParameter('ambientRadiance', pynari.FLOAT32, 1.)
    renderer.setParameter('background', pynari.ARRAY, bg_gradient)
    
    if pixel_samples is None:
        if pynari.has_cuda_capable_gpu():
            pixel_samples = 1024
        else:
            pixel_samples = 16
    
    renderer.setParameter('pixelSamples', pynari.INT32, pixel_samples)
    renderer.commitParameters()

    frame = device.newFrame()
    frame.setParameter('size', pynari.uint2, fb_size)
    frame.setParameter('channel.color', pynari.DATA_TYPE, pynari.UFIXED8_RGBA_SRGB)
    frame.setParameter('renderer', pynari.OBJECT, renderer)
    frame.setParameter('camera', pynari.OBJECT, camera)
    frame.setParameter('world', pynari.OBJECT, world)
    frame.commitParameters()
    
    return {'renderer': renderer, 'frame': frame}

def render_volume(volume_input, volume_dims, W, H,
                 camera_config, tf, bg_color, pixel_samples=None, render_setup=None):
    fb_size = (W, H)
    volume_data = volume_input.copy()
    # 确保传输函数格式正确
    if tf.ndim == 2 and tf.shape[1] == 4:  # 如果是(N,4)格式
        # 将其展平为一维数组
        tf = tf.reshape(-1)
    
    # 设置体积渲染器
    
    # 创建设备
    device = pynari.newDevice('default')
    
    # 计算单元格大小和原点
    max_dim = max(volume_dims)
    cellSize = (2/max_dim, 2/max_dim, 2/max_dim)
    
    actual_x = volume_dims[0] * cellSize[0]
    actual_y = volume_dims[1] * cellSize[1]
    actual_z = volume_dims[2] * cellSize[2]
    origin = (-actual_x/2, -actual_y/2, -actual_z/2)
    
    # 创建ANARI对象
    structured_data = device.newArray(pynari.float, volume_data)
    spatial_field = device.newSpatialField('structuredRegular')
    spatial_field.setParameter('origin', pynari.float3, origin)
    spatial_field.setParameter('spacing', pynari.float3, cellSize)
    spatial_field.setParameter('data', pynari.ARRAY3D, structured_data)
    spatial_field.commitParameters()

    xf_array = device.newArray(pynari.float4, tf)
    volume = device.newVolume('transferFunction1D')
    volume.setParameter('color', pynari.ARRAY, xf_array)
    volume.setParameter('value', pynari.SPATIAL_FIELD, spatial_field)
    volume.setParameter('unitDistance', pynari.FLOAT32, 10.)
    volume.commitParameters()

    world = device.newWorld()
    volume_array = device.newArray(pynari.VOLUME, [volume])
    world.setParameter('volume', pynari.ARRAY3D, volume_array)

    # 设置光照
    light = device.newLight('directional')
    light.setParameter('direction', pynari.float3, (1., -1., -1.))
    light.commitParameters()
    light_array = device.newArray(pynari.LIGHT, [light])
    world.setParameter('light', pynari.ARRAY1D, light_array)
    world.commitParameters()
    
    # 设置相机
    center = (0, 0, 0)
    look_from = spherical_to_cartesian(
        camera_config['pitch'], 
        camera_config['yaw'], 
        camera_config['distance'], 
        center
    )
    look_at=center
    look_up=(0., 1., 0.)
    fovy = 40.
    camera = device.newCamera('perspective')
    camera.setParameter('aspect', pynari.FLOAT32, fb_size[0]/fb_size[1])
    camera.setParameter('position', pynari.FLOAT32_VEC3, look_from)
    
    direction = [look_at[0] - look_from[0],
                 look_at[1] - look_from[1],
                 look_at[2] - look_from[2]]
    camera.setParameter('direction', pynari.float3, direction)
    camera.setParameter('up', pynari.float3, look_up)
    camera.setParameter('fovy', pynari.float, fovy * math.pi / 180)
    camera.commitParameters()
    
    # 设置渲染器
    # if render_setup is None:
    
    bg_color = bg_color.strip("() ").split(',')
    bg_color = np.array(bg_color, dtype=np.float32)
    bg_color = np.concatenate((bg_color, [1]))
    bg_color = np.clip(bg_color, 0, 1)
    bg_values = np.array((bg_color,bg_color), dtype=np.float32).reshape((2,1,4))
    bg_gradient = device.newArray(pynari.float4, bg_values)

    renderer = device.newRenderer('default')
    renderer.setParameter('ambientRadiance', pynari.FLOAT32, 1.)
    renderer.setParameter('background', pynari.ARRAY, bg_gradient)
    
    if pixel_samples is None:
        if pynari.has_cuda_capable_gpu():
            pixel_samples = 1024
        else:
            pixel_samples = 16
    
    renderer.setParameter('pixelSamples', pynari.INT32, pixel_samples)
    renderer.commitParameters()

    frame = device.newFrame()
    frame.setParameter('size', pynari.uint2, fb_size)
    frame.setParameter('channel.color', pynari.DATA_TYPE, pynari.UFIXED8_RGBA_SRGB)
    frame.setParameter('renderer', pynari.OBJECT, renderer)
    frame.setParameter('camera', pynari.OBJECT, camera)
    frame.setParameter('world', pynari.OBJECT, world)
    frame.commitParameters()
    
    # 渲染
    frame.render()
    fb_color = frame.get('channel.color')
    pixels = np.array(fb_color)

    im = Image.fromarray(pixels)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.convert('RGB')

    # 释放资源
    spatial_field.release()
    structured_data.release()
    volume.release()
    xf_array.release()
    volume_array.release()
    world.release()
    light_array.release()
    light.release()
    camera.release()
    bg_gradient.release()
    renderer.release()
    frame.release()
    
    return im


# def render_volume(volume_data, volume_dims, W, H,
#                  camera_config, tf, bg_color, pixel_samples=None, render_setup=None):
#     """
#     渲染体积数据的简单接口
    
#     Args:
#         volume_data: 体积数据numpy数组
#         volume_dims: 体积维度 (width, height, depth)
#         output_path: 输出文件路径，None为显示图像
#         W,H: 帧的长宽
#         camera_config: 相机配置字典 {'pitch': 弧度, 'yaw': 弧度, 'distance': 距离}
#         tf: 传输函数
#         bg_color: 背景颜色，输入为字符串"(r,g,b)"格式
#         pixel_samples: 像素采样数
    
#     Returns:
#         numpy数组: 渲染后的像素数据
#     """
#     fb_size = (W, H)
    
#     # 确保传输函数格式正确
#     if tf.ndim == 2 and tf.shape[1] == 4:  # 如果是(N,4)格式
#         # 将其展平为一维数组
#         tf = tf.reshape(-1)
    
#     # 设置体积渲染器
#     volume_setup = setup_volume_renderer(volume_data, volume_dims, tf)
#     device = volume_setup['device']
#     world = volume_setup['world']
    
#     # 设置相机
#     center = volume_setup['volume_info']['center']
#     look_from = spherical_to_cartesian(
#         camera_config['pitch'], 
#         camera_config['yaw'], 
#         camera_config['distance'], 
#         center
#     )
#     camera = setup_camera(
#         device=device, 
#         W=fb_size[0], 
#         H=fb_size[1], 
#         look_from=look_from, 
#         look_at=center, 
#         look_up=(0., 1., 0.)
#     )
    
#     # 设置渲染器
#     # if render_setup is None:
#     render_setup = setup_renderer(device, camera, world, fb_size, bg_color, pixel_samples)
#     renderer = render_setup['renderer']
#     frame = render_setup['frame']
    
#     # 渲染
#     frame.render()
#     fb_color = frame.get('channel.color')
#     pixels = np.array(fb_color)

#     im = Image.fromarray(pixels)
#     # im = im.transpose(Image.FLIP_TOP_BOTTOM)
#     im = im.convert('RGB')
    
#     return im
    

def create_rotation_video(volume_data, volume_dims,output_path='rotation_video.mp4',
                         fps=30, duration=8, W=1600, H=800,
                         radius=3.0, tf=None, pixel_samples=None, render_setup=None):
    """
    创建体积数据的旋转视频
    
    Args:
        volume_data: 体积数据numpy数组
        volume_dims: 体积维度 (width, height, depth)
        fps: 帧率
        duration: 视频时长（秒）
        W,H: 帧的长宽
        radius: 相机旋转半径
        tf: 传输函数
        pixel_samples: 像素采样数
    
    Returns:
        str: 输出视频路径
    """
    fb_size = (W, H)
    # 设置体积渲染器
    volume_setup = setup_volume_renderer(volume_data, volume_dims, tf)
    device = volume_setup['device']
    world = volume_setup['world']
    center = volume_setup['volume_info']['center']
    
    
    # 设置初始相机
    camera = setup_camera(device, W, H, (radius, 0.5, 0), center, (0., 1., 0.))


    # 设置渲染器
    render_setup = setup_renderer(device, camera, world, fb_size, pixel_samples)
    renderer = render_setup['renderer']
    frame = render_setup['frame']

    frame.setParameter('renderer', pynari.OBJECT, renderer)
    frame.setParameter('camera', pynari.OBJECT, camera)
    frame.setParameter('world', pynari.OBJECT, world)
    frame.setParameter('channel.color', pynari.DATA_TYPE, pynari.UFIXED8_RGBA_SRGB)
    frame.commitParameters()
    
    # 视频设置
    total_frames = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, fb_size)
    
    print(f"开始生成旋转视频...")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps} FPS")
    print(f"  视频时长: {duration} 秒")
    print(f"  输出文件: {output_path}")
    
    for frame_idx in tqdm(range(total_frames), desc="渲染帧"):
        # 计算当前帧的旋转角度
        angle = (frame_idx / total_frames) * 2 * math.pi
        
        # 计算相机位置（绕Y轴旋转）
        camera_x = center[0] + radius * math.sin(angle)
        camera_z = center[2] + radius * math.cos(angle)
        camera_y = center[1] + 0.5
        
        look_from = (camera_x, camera_y, camera_z)
        
        # 更新相机参数
        camera.setParameter('position', pynari.FLOAT32_VEC3, look_from)
        direction = [center[0] - look_from[0],
                    center[1] - look_from[1],
                    center[2] - look_from[2]]
        camera.setParameter('direction', pynari.float3, direction)
        camera.commitParameters()
        
        # 渲染当前帧
        frame.render()
        fb_color = frame.get('channel.color')
        pixels = np.array(fb_color)
        
        # 转换为OpenCV格式（BGR）
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        pixels_bgr = cv2.flip(pixels_bgr, 0)
        
        # 写入视频帧
        video_writer.write(pixels_bgr)
    
    video_writer.release()
    print(f"✓ 视频已保存到: {output_path}")
    return output_path

# # 简化的接口函数
# def render_raw_file(filepath, output_path=None, video=False, **kwargs):
#     """
#     直接从RAW文件渲染的便捷函数
#     需要配合volume_loader使用
    
#     Args:
#         filepath: RAW文件路径
#         output_path: 输出路径
#         video: 是否生成视频
#         **kwargs: 其他参数
#     """
#     try:
#         from volume_loader import load_volume_simple
        
#         # 加载体积数据
#         result = load_volume_simple(filepath)
#         volume_data = result['data']
#         volume_dims = result['original_dims']
        
#         if video:
#             return create_rotation_video(volume_data, volume_dims, output_path, **kwargs)
#         else:
#             return render_volume(volume_data, volume_dims, output_path, **kwargs)
            
#     except ImportError:
#         raise ImportError("需要volume_loader模块来加载RAW文件")