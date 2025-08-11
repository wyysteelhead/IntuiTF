"""
This file is taken from the DiffDVR repository (https://github.com/shamanDevel/DiffDVR).
It contains the core configuration and settings for the DiffDVR library, including:
- Volume rendering parameters
- Transfer function settings
- Camera and viewport configurations
- Optimization settings
"""

import torch
import numpy as np
import json
import os
from typing import Optional, NamedTuple

from diffdvr.utils import make_real3
from genetic_optimize.utils.dtype_utils import get_renderer_dtypes
renderer_dtype_torch, renderer_dtype_np = get_renderer_dtypes("diffdvr")
import pyrenderer
from genetic_optimize.utils.file_utils import get_project_root

"""
Loads settings from .json file exported by the GUI
"""

class Settings:

    def __init__(self, file):
        self._filepath = get_project_root()
        with open(file) as fb:
            self._data = json.load(fb)
        if self._data["version"] != 2:
            raise Exception("incorrect file version, expected 2 but got "+self._data["version"])

    # Dataset handling

    def is_synthetic_dataset(self):
        return self._data["dataset"]["syntheticDataset"] >= 0

    def get_synthetic_dataset_type(self) -> pyrenderer.ImplicitEquation:
        assert self.is_synthetic_dataset()
        return pyrenderer.ImplicitEquation(int(self._data["dataset"]["syntheticDataset"]))

    def get_dataset_name(self):
        if self.is_synthetic_dataset():
            t = self.get_synthetic_dataset_type()
            return t.name
        else:
            path = self._data["dataset"]["cvol_file"]
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            abs_path = os.path.abspath(os.path.join(project_root, path))
            return os.path.split(abs_path)[-1]

    def load_dataset(self, resolution : Optional[int] = None) -> pyrenderer.Volume:
        """
        Loads the dataset specified in the settings file
        :param resolution: if it is a synthetic dataset, use this resolution
          instead of the one specified in the settings
        :return: the volume, still only on the CPU
        """

        if self.is_synthetic_dataset():
            if resolution is None:
                resolution = 1 << int(self._data["dataset"]["syntheticDatasetResolutionPower"])
            t = self.get_synthetic_dataset_type()
            return pyrenderer.Volume.create_implicit(t, resolution)
        else:
            path = self._data["dataset"]["cvol_file"].replace("\\", "/")
            if not os.path.isabs(path):
                path = os.path.abspath(os.path.join(self._filepath, path))
                print("convert relative path to absolute path, load", path)
            return pyrenderer.Volume(path)
        
    def load_gradient(self, resolution : Optional[int] = None) -> pyrenderer.Volume:
        """
        Loads the gradien of dataset specified in the settings file
        :param resolution: if it is a synthetic dataset, use this resolution
          instead of the one specified in the settings
        :return: the volume, still only on the CPU
        """

        path = self._data["dataset"]["cvol_file"].replace("\\", "/")
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(self._filepath, path))
            print("convert relative path to absolute path, load", path)
        # 修改文件名，添加 "grad"
        base, ext = os.path.splitext(path)  # 拆分路径和扩展名
        if ext == ".cvol":
            path = f"{base}grad{ext}"
        if os.path.exists(path):
            return pyrenderer.Volume(path)
        else:
            #TODO add gradient calculation
            return None

    class CameraConfig(NamedTuple):
        pitch_radians : float
        yaw_radians : float
        fov_y_radians : float
        center : np.ndarray # shape=(3,)
        distance : float
        orientation : pyrenderer.Orientation

    def get_camera(self) -> CameraConfig:
        """
        Loads the camera configuration.
        The results can be used for parametrizations.CameraOnASphere
        :return:
        """
        c = self._data["camera"]
        return Settings.CameraConfig(
            np.deg2rad(c["pitch"]),
            np.deg2rad(c["yaw"]),
            np.deg2rad(c["fov"]),
            np.array(c["lookAt"]),
            np.array(c["distance"]),
            pyrenderer.Orientation(int(c["orientation"]))
        )

    def get_tf_points(self,
                      min_density : Optional[float] = None,
                      max_density : Optional[float] = None,
                      opacity_scaling : Optional[float] = None,
                      purge_zero_regions : bool = True):
        """
        Returns the list of TF points, to be converted to a tensor either via
        pyrenderer.TFUtils.get_piecewise_tensor or pyrenderer.TFUtils.get_texture_tensor
        :return: a list of control points
        """
        if min_density is None:
            min_density = self._data["tfEditor"]["minDensity"]
        if max_density is None:
            max_density = self._data["tfEditor"]["maxDensity"]
        if opacity_scaling is None:
            opacity_scaling = self._data["tfEditor"]["opacityScaling"]

        key = "editor" if "editor" in self._data["tfEditor"] else "editorLinear"
        g = self._data["tfEditor"][key]
        return pyrenderer.TFUtils.assemble_from_settings(
            [pyrenderer.real3(v[0], v[1], v[2]) for v in g["colorAxis"]],
            g["densityAxisColor"],
            g["opacityAxis"],
            g["densityAxisOpacity"],
            min_density, max_density, opacity_scaling,
            purge_zero_regions
        )

    def get_tf_points_texture(self,
                           min_density: Optional[float] = None,
                           max_density: Optional[float] = None,
                           opacity_scaling: Optional[float] = None):
        if min_density is None:
            min_density = self._data["tfEditor"]["minDensity"]
        if max_density is None:
            max_density = self._data["tfEditor"]["maxDensity"]
        if opacity_scaling is None:
            opacity_scaling = self._data["tfEditor"]["opacityScaling"]

        g = self._data["tfEditor"]['editorTexture']
        colorAxis = [pyrenderer.real3(v[0], v[1], v[2]) for v in g["colorAxis"]]
        densityAxisColor = g["densityAxisColor"]
        opacityAxis = g["opacities"]
        densityAxisOpacity = list(np.linspace(0, 1, len(opacityAxis), endpoint=True))
        return pyrenderer.TFUtils.assemble_from_settings(
            colorAxis, densityAxisColor, opacityAxis, densityAxisOpacity,
            min_density, max_density, opacity_scaling, False
        )

    def get_gaussian_tensor(self,
                            min_density: Optional[float] = None,
                            max_density: Optional[float] = None,
                            opacity_scaling: Optional[float] = None):
        """
        Returns the TF tensor of the gaussian transfer function
        """
        if min_density is None:
            min_density = self._data["tfEditor"]["minDensity"]
        if max_density is None:
            max_density = self._data["tfEditor"]["maxDensity"]
        if opacity_scaling is None:
            opacity_scaling = self._data["tfEditor"]["opacityScaling"]

        g = self._data["tfEditor"]["editorGaussian"]
        R = len(g)
        tf = np.empty((1, R, 6), dtype=renderer_dtype_np)
        for r in range(R):
            tf[0][r][0] = g[r][0] # red
            tf[0][r][1] = g[r][1] # green
            tf[0][r][2] = g[r][2] # blue
            tf[0][r][3] = g[r][3] * opacity_scaling # opacity
            tf[0][r][4] = min_density + g[r][4] * (max_density - min_density) # mean
            tf[0][r][5] = g[r][5] * (max_density-min_density) # variance
        return torch.from_numpy(tf)

    def get_stepsize(self):
        return self._data["renderer"]["stepsize"]

def setup_default_settings(
        volume : pyrenderer.Volume,
        gradient : pyrenderer.Volume,
        screen_width : int, screen_height : int,
        step_size : float = 0.1,
        run_on_cuda : bool = True,
        copy_to_gpu : bool = False,
        volume_data = None,
        gradient_data = None,):
    if copy_to_gpu is True:
        volume_data = volume.getDataGpu(0) if run_on_cuda else volume.getDataCpu(0)
        if gradient is not None:
            gradient_data = gradient.getDataGpu(0) if run_on_cuda else gradient.getDataCpu(0)
        else:
            gradient_data = volume_data.clone()

    # copied from visualizer.cpp -> setupRendererArgs()
    box_size = np.array([volume.world_size.x, volume.world_size.y, volume.world_size.z])
    resolution = np.array([volume.resolution.x, volume.resolution.y, volume.resolution.z])
    voxel_size = box_size / (resolution - np.array([1,1,1]))
    box_min = (-box_size*0.5) - (voxel_size*0.5)

    settings = pyrenderer.RendererInputs()
    settings.volume = volume_data
    settings.gradient = gradient_data
    settings.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
    settings.screen_size = pyrenderer.int2(screen_width, screen_height)
    settings.box_min = make_real3(box_min)
    settings.box_size = make_real3(box_size)
    settings.step_size = step_size / resolution[0]
    settings.blend_mode = pyrenderer.BlendMode.BeerLambert

    return settings


def setup_settings_from_settings(settings: pyrenderer.RendererInputs):

    new_settings = pyrenderer.RendererInputs()
    new_settings.volume = settings.volume
    new_settings.gradient = settings.gradient
    new_settings.volume_filter_mode = settings.volume_filter_mode
    new_settings.screen_size = settings.screen_size
    new_settings.box_min = settings.box_min
    new_settings.box_size = settings.box_size
    new_settings.step_size = settings.step_size
    new_settings.blend_mode = settings.blend_mode

    return new_settings
