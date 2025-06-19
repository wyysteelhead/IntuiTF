"""
This file is taken from the DiffDVR repository (https://github.com/shamanDevel/DiffDVR).
It contains utilities for transfer function parametrization, including:
- Transfer function parameter handling
- Parameter space transformations
- Optimization-related utilities
"""

import torch
import numpy as np
import pyrenderer

class CameraOnASphere(torch.nn.Module):
    """
    Parametrization of a camera on a bounding sphere.
    Parameters:
      - orientation (pyrenderer.Orientation) fixed in the constructor
      - center (B*3) vector, specified in self.forward()
      - yaw (B*1) in radians, specified in self.forward()
      - pitch (B*1) in radians, specified in self.forward()
      - distance (B*1) scalar, specified in self.forward()
    """
    ZoomBase = 1.1

    def __init__(self, orientation : pyrenderer.Orientation):
        super().__init__()
        self._orientation = orientation

    def forward(self, center, yaw, pitch, distance):
        return pyrenderer.Camera.viewport_from_sphere(
            center, yaw, pitch, distance, self._orientation)

    @staticmethod
    def random_points(N : int, min_zoom = 6, max_zoom = 7, center = None):
        """
        Generates N random points on the sphere
        :param N: the number of points
        :param min_zoom: the minimal zoom factor
        :param max_zoom: the maximal zoom factor
        :param center: the center of the sphere. If None, [0,0,0] is used
        :return: a tuple with tensors (center, yaw, pitch, distance)
        """

        if center is None:
            center = [0,0,0]

        # random points on the unit sphere
        vec = np.random.randn(N, 3)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        # convert to pitch and yaw
        pitch = np.arccos(vec[:,2])
        yaw = np.arctan2(vec[:,1], vec[:,0])

        # sample distances
        dist = np.random.uniform(min_zoom, max_zoom, (N,))
        dist = np.power(CameraOnASphere.ZoomBase, dist)

        # to pytorch tensors
        dtype = torch.float64 if pyrenderer.use_double_precision() else torch.float32
        center = torch.tensor([center]*N, dtype=dtype)
        yaw = torch.from_numpy(yaw).to(dtype=dtype).unsqueeze(1)
        pitch = torch.from_numpy(pitch).to(dtype=dtype).unsqueeze(1)
        dist = torch.from_numpy(dist).to(dtype=dtype).unsqueeze(1)

        return center, yaw, pitch, dist
