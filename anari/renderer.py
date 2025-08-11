import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Try to import pynari
try:
    import pynari
except ImportError:
    print("Warning: Unable to import pynari, please ensure it is properly installed")

def spherical_to_cartesian(pitch, yaw, distance, center):
    """Convert spherical coordinates to cartesian coordinates"""
    x = distance * math.cos(pitch) * math.sin(yaw)
    y = distance * math.sin(pitch)
    z = distance * math.cos(pitch) * math.cos(yaw)
    return (center[0] + x, center[1] + y, center[2] + z)

def setup_volume_renderer(volume_data, volume_dims, tf,
                         center_at_origin=True, device=None):
    """
    Setup volume renderer
    
    Args:
        volume_data: Volume data numpy array, shape (depth, height, width)
        volume_dims: Original volume dimensions (width, height, depth)
        tf: Transfer function numpy array, None for automatic selection
        center_at_origin: Whether to center the volume at the origin
        device: ANARI device, None to create new device
    
    Returns:
        dict: {
            'device': ANARI device,
            'world': World object,
            'volume': Volume object,
            'spatial_field': Spatial field object,
            'volume_info': Volume information dictionary
        }
    """
    if pynari is None:
        raise ImportError("pynari not installed, cannot use rendering functionality")
    
    # Create device
    if device is None:
        device = pynari.newDevice('default')
    
    # Calculate cell size and origin
    max_dim = max(volume_dims)
    cellSize = (2/max_dim, 2/max_dim, 2/max_dim)
    
    if center_at_origin:
        # Center setting
        actual_x = volume_dims[0] * cellSize[0]
        actual_y = volume_dims[1] * cellSize[1]
        actual_z = volume_dims[2] * cellSize[2]
        origin = (-actual_x/2, -actual_y/2, -actual_z/2)
    else:
        origin = (-1, -1, -1)
    
    # Create ANARI objects
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

    # Setup lighting
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
    Setup camera
    
    Args:
        device: ANARI device
        fb_size: Frame buffer size (width, height)
        look_from: Camera position (x, y, z)
        look_at: Look at target (x, y, z)
        look_up: Up direction (x, y, z)
        fovy: Field of view angle (degrees)
    
    Returns:
        camera: Camera object
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
    Setup renderer
    
    Args:
        device: ANARI device
        fb_size: Frame buffer size (width, height)
        bg_color: Background color, input as string "(r,g,b)" format
        pixel_samples: Pixel sample count, None for automatic selection
    
    Returns:
        dict: {'renderer': Renderer, 'frame': Frame object}
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
    # Ensure transfer function format is correct
    if tf.ndim == 2 and tf.shape[1] == 4:  # If (N,4) format
        # Flatten to 1D array
        tf = tf.reshape(-1)
    
    # Setup volume renderer
    
    # Create device
    device = pynari.newDevice('default')
    
    # Calculate cell size and origin
    max_dim = max(volume_dims)
    cellSize = (2/max_dim, 2/max_dim, 2/max_dim)
    
    actual_x = volume_dims[0] * cellSize[0]
    actual_y = volume_dims[1] * cellSize[1]
    actual_z = volume_dims[2] * cellSize[2]
    origin = (-actual_x/2, -actual_y/2, -actual_z/2)
    
    # Create ANARI objects
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

    # Setup lighting
    light = device.newLight('directional')
    light.setParameter('direction', pynari.float3, (1., -1., -1.))
    light.commitParameters()
    light_array = device.newArray(pynari.LIGHT, [light])
    world.setParameter('light', pynari.ARRAY1D, light_array)
    world.commitParameters()
    
    # Setup camera
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
    
    # Setup renderer
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
    
    # Render
    frame.render()
    fb_color = frame.get('channel.color')
    pixels = np.array(fb_color)

    im = Image.fromarray(pixels)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.convert('RGB')

    # Release resources
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
#     Simple interface for rendering volume data
    
#     Args:
#         volume_data: Volume data numpy array
#         volume_dims: Volume dimensions (width, height, depth)
#         output_path: Output file path, None to display image
#         W,H: Frame width and height
#         camera_config: Camera configuration dict {'pitch': radians, 'yaw': radians, 'distance': distance}
#         tf: Transfer function
#         bg_color: Background color, input as string "(r,g,b)" format
#         pixel_samples: Pixel sample count
    
#     Returns:
#         numpy array: Rendered pixel data
#     """
#     fb_size = (W, H)
    
#     # Ensure transfer function format is correct
#     if tf.ndim == 2 and tf.shape[1] == 4:  # If (N,4) format
#         # Flatten to 1D array
#         tf = tf.reshape(-1)
    
#     # Setup volume renderer
#     volume_setup = setup_volume_renderer(volume_data, volume_dims, tf)
#     device = volume_setup['device']
#     world = volume_setup['world']
    
#     # Setup camera
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
    
#     # Setup renderer
#     # if render_setup is None:
#     render_setup = setup_renderer(device, camera, world, fb_size, bg_color, pixel_samples)
#     renderer = render_setup['renderer']
#     frame = render_setup['frame']
    
#     # Render
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
    Create rotation video of volume data
    
    Args:
        volume_data: Volume data numpy array
        volume_dims: Volume dimensions (width, height, depth)
        fps: Frame rate
        duration: Video duration (seconds)
        W,H: Frame width and height
        radius: Camera rotation radius
        tf: Transfer function
        pixel_samples: Pixel sample count
    
    Returns:
        str: Output video path
    """
    fb_size = (W, H)
    # Setup volume renderer
    volume_setup = setup_volume_renderer(volume_data, volume_dims, tf)
    device = volume_setup['device']
    world = volume_setup['world']
    center = volume_setup['volume_info']['center']
    
    
    # Setup initial camera
    camera = setup_camera(device, W, H, (radius, 0.5, 0), center, (0., 1., 0.))


    # Setup renderer
    render_setup = setup_renderer(device, camera, world, fb_size, pixel_samples)
    renderer = render_setup['renderer']
    frame = render_setup['frame']

    frame.setParameter('renderer', pynari.OBJECT, renderer)
    frame.setParameter('camera', pynari.OBJECT, camera)
    frame.setParameter('world', pynari.OBJECT, world)
    frame.setParameter('channel.color', pynari.DATA_TYPE, pynari.UFIXED8_RGBA_SRGB)
    frame.commitParameters()
    
    # Video settings
    total_frames = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, fb_size)
    
    print(f"Starting rotation video generation...")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame rate: {fps} FPS")
    print(f"  Video duration: {duration} seconds")
    print(f"  Output file: {output_path}")
    
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        # Calculate rotation angle for current frame
        angle = (frame_idx / total_frames) * 2 * math.pi
        
        # Calculate camera position (rotate around Y axis)
        camera_x = center[0] + radius * math.sin(angle)
        camera_z = center[2] + radius * math.cos(angle)
        camera_y = center[1] + 0.5
        
        look_from = (camera_x, camera_y, camera_z)
        
        # Update camera parameters
        camera.setParameter('position', pynari.FLOAT32_VEC3, look_from)
        direction = [center[0] - look_from[0],
                    center[1] - look_from[1],
                    center[2] - look_from[2]]
        camera.setParameter('direction', pynari.float3, direction)
        camera.commitParameters()
        
        # Render current frame
        frame.render()
        fb_color = frame.get('channel.color')
        pixels = np.array(fb_color)
        
        # Convert to OpenCV format (BGR)
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        pixels_bgr = cv2.flip(pixels_bgr, 0)
        
        # Write video frame
        video_writer.write(pixels_bgr)
    
    video_writer.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path

# # Simplified interface function
# def render_raw_file(filepath, output_path=None, video=False, **kwargs):
#     """
#     Convenience function for rendering directly from RAW files
#     Requires use with volume_loader
    
#     Args:
#         filepath: RAW file path
#         output_path: Output path
#         video: Whether to generate video
#         **kwargs: Other parameters
#     """
#     try:
#         from volume_loader import load_volume_simple
        
#         # Load volume data
#         result = load_volume_simple(filepath)
#         volume_data = result['data']
#         volume_dims = result['original_dims']
        
#         if video:
#             return create_rotation_video(volume_data, volume_dims, output_path, **kwargs)
#         else:
#             return render_volume(volume_data, volume_dims, output_path, **kwargs)
            
#     except ImportError:
#         raise ImportError("Requires volume_loader module to load RAW files")