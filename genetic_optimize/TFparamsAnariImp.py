# The anari module is not ready yet, we are still dealing with parallel rendering process
import numpy as np
import torch
from PIL import Image
from genetic_optimize.TFparamsBase import TFparamsBase
from genetic_optimize.states.bound import Bound
from genetic_optimize.utils.image_utils import image_to_base64_pil, add_red_border
import pynari
from anari.settings import setup_default_settings, setup_settings_from_settings, AnariSettings
import tempfile
import io
import math
from anari.renderer import render_volume


class TFparamsAnariImp(TFparamsBase):
    """
    Transfer function parameters class implemented using Anari (pynari) renderer
    """
    
    global_inputs = None  # Class variable, stores global render settings
    
    def __init__(self, id: int, bound: Bound=None, volume: np.array = None, gradient: np.array = None, step_size = None, initial_rating=1600, W=512, H=512, bg_color=None, device="cuda", renderer_dtype_np = np.float64, tfparams=None, setInputs=False):
        super().__init__(id, bound, initial_rating, W, H, bg_color, device, renderer_dtype_np, tfparams)
        
        # Initialize renderer settings
        if setInputs and bound is not None:
            self.__initialize_render_inputs(volume, gradient, step_size)
        elif setInputs and tfparams is not None:
            self.__initialize_settings(setInputs=setInputs)
            
    def __spherical_to_cartesian(self, pitch, yaw, distance, center=(0,0,0)):
        """Convert spherical coordinates to Cartesian coordinates"""
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.sin(pitch)
        z = distance * math.cos(pitch) * math.cos(yaw)
        return (center[0] + x, center[1] + y, center[2] + z)
            
    def __initialize_render_inputs(self, volume, gradient, step_size):
        """Initialize renderer input parameters"""
        if volume is None:
            raise ValueError("Volume data must be provided")
            
        # If global settings don't exist, create new settings
        if TFparamsAnariImp.global_inputs is None:
            TFparamsAnariImp.global_inputs = setup_default_settings(
                volume_data=volume,
                screen_width=self.W,
                screen_height=self.H
            )
        
        # Create instance settings from global settings
        self.inputs = setup_settings_from_settings(TFparamsAnariImp.global_inputs)
        
    def __initialize_settings(self, setInputs=True):
        """Initialize instance settings from global settings"""
        if setInputs and TFparamsAnariImp.global_inputs is not None:
            self.inputs = setup_settings_from_settings(TFparamsAnariImp.global_inputs)
            
    def _get_copy(self):
        """Override parent class abstract method"""
        copy_of_self = TFparamsAnariImp(id=self.id, tfparams=self, setInputs=True)
        return copy_of_self
        
    def render_image(self, device="cuda", bg_color=None, force_render=False, 
                    setImage=True, addRedBorder=False, tf=None, transparent=False):
        """Render image from single viewpoint"""
        if not hasattr(self, 'inputs') or not hasattr(self.inputs, 'device'):
            raise RuntimeError("Renderer not initialized, please call __initialize_render_inputs first")
        
        if self.image is not None and not force_render:
            return self.image

        # Set camera configuration
        camera_config = {
            'pitch': 3.5,  # Approximately 90 degree elevation
            'yaw': 2.2,    # 0 degree yaw angle
            'distance': 10.0
        }

        # If no transfer function is provided, use current transfer function
        if tf is None:
            tf = self.get_tf().cpu().numpy()[0]  # Remove batch dimension
        if bg_color is None:
            bg_color = self.bg_color

        # Use render_volume function from renderer.py
        im = render_volume(
            volume_input=self.inputs.volume_data,
            volume_dims=self.inputs.volume_dims,
            W=self.W,
            H=self.H,
            camera_config=camera_config,
            tf=tf,
            bg_color=bg_color,
            pixel_samples=1024 if pynari.has_cuda_capable_gpu() else 16
        )

        if addRedBorder:
            im = add_red_border(im)
            
        if setImage:
            self.image = im
            
        return im
    
    def render_image_on_sphere(self, num_views=300, device="cuda", bg_color=None):
        """Render multiple viewpoint images on sphere"""
        if not hasattr(self, 'inputs') or not hasattr(self.inputs, 'device'):
            raise RuntimeError("Renderer not initialized, please call __initialize_render_inputs first")
            
        images = []
        radius = 3.4  # Camera distance
        
        # Generate uniformly distributed points on sphere
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        for i in range(num_views):
            y = 1 - (i / float(num_views - 1)) * 2  # y from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # Circle radius at current y position
            
            theta = phi * i  # Azimuth angle
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            # Calculate spherical coordinates
            pitch = np.arcsin(y)
            yaw = np.arctan2(x, z)
            
            # Set camera configuration for current viewpoint
            camera_config = {
                'pitch': pitch,
                'yaw': yaw,
                'distance': radius
            }

            # Use render_volume function from renderer.py
            im = render_volume(
                volume_data=self.inputs.volume_data,
                volume_dims=self.inputs.volume_dims,
                W=self.W,
                H=self.H,
                camera_config=camera_config,
                tf=self.get_tf().cpu().numpy()[0],  # Use current transfer function
                pixel_samples=1024 if pynari.has_cuda_capable_gpu() else 16
            )
            
            images.append(im)
            
        return images