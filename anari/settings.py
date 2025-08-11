"""
This file contains the core configuration and settings for the pynari library, following
the structure of diffdvr/settings.py but delegating rendering functionality to renderer.py
"""

import os
import json
import numpy as np
import pynari
from typing import Optional, NamedTuple
from .process_volume import load_and_process_volume
from .renderer import setup_renderer, setup_camera

class CameraConfig(NamedTuple):
    """Camera configuration data structure"""
    pitch_radians: float
    yaw_radians: float
    fov_y_radians: float
    center: np.ndarray  # shape=(3,)
    distance: float

class AnariSettings:
    """ANARI renderer settings and configuration management"""
    def __init__(self, file=None):
        # Initialize with default values
        self.volume_data = None
        self.volume_dims = None
        self.screen_size = (512, 512)  # default size
        self.pixel_samples = 1024 if pynari.has_cuda_capable_gpu() else 16
        self.device = pynari.newDevice('default')
        
        if file is not None:
            self._filepath = os.path.split(file)[0]
            with open(file) as fb:
                self._data = json.load(fb)
            if self._data["version"] != 2:
                raise Exception("incorrect file version, expected 2 but got "+str(self._data["version"]))
            
            # Load settings from file
            self.screen_size = (
                self._data["renderer"].get("width", 512),
                self._data["renderer"].get("height", 512)
            )

    def load_dataset(self):
        """Load volume dataset from file"""
        if not hasattr(self, '_data'):
            raise ValueError("No configuration file loaded")
            
        path = self._data["dataset"]["raw_file"].replace("\\", "/")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_path = os.path.abspath(os.path.join(project_root, path))
        path = os.path.split(abs_path)[-1]
        
        print("Loading volume from:", abs_path)
        
        data, dims, spacing = load_and_process_volume(
            abs_path,
            threshold_ratio=0.1,
            normalize_range=(0.2, 1.0)
        )
        self.volume_data = data
        self.volume_dims = dims
        return data

    def get_camera_config(self) -> dict:
        """Get camera configuration from settings"""
        if not hasattr(self, '_data'):
            # Return default camera config
            return {
                'pitch': 1.57,  # 90 degrees
                'yaw': 0.0,
                'distance': 3.4
            }
            
        c = self._data["camera"]
        return {
            'pitch': float(c["pitch"]),
            'yaw': float(c["yaw"]),
            'distance': float(c["distance"])
        }

    def get_screen_size(self):
        """Get the screen dimensions for rendering"""
        return self.screen_size

    def get_stepsize(self):
        """Get the step size for rendering"""
        return self._data["renderer"]["stepsize"]

def setup_default_settings(
        volume_data: np.ndarray,
        screen_width: int = 512,
        screen_height: int = 512) -> AnariSettings:
    """
    Setup default ANARI renderer settings
    
    Args:
        volume_data: The volume data as numpy array
        screen_width: Width of the output image
        screen_height: Height of the output image
    
    Returns:
        AnariSettings: The initialized settings object
    """
    settings = AnariSettings()
    settings.volume_data = volume_data
    settings.volume_dims = volume_data.shape
    settings.screen_size = (screen_width, screen_height)
    return settings

def setup_settings_from_settings(settings: AnariSettings) -> AnariSettings:
    """
    Create a new settings object from an existing one
    
    Args:
        settings: The source settings object
    
    Returns:
        AnariSettings: A new settings object with copied values
    """
    new_settings = AnariSettings()
    new_settings.volume_data = settings.volume_data
    new_settings.volume_dims = settings.volume_dims
    new_settings.screen_size = settings.screen_size
    new_settings.pixel_samples = settings.pixel_samples
    return new_settings 