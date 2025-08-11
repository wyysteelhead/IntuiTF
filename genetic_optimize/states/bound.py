from types import SimpleNamespace
import numpy as np
# from tools.cameraGenerator import generateByFibonacciSphere # Magic spell, removing this causes pyrenderer import failure
# from module import Module
import json

class Bound():
    """
    A class to manage and store bound for camera, opacity, and color configurations.

    This class reads a configuration file, initializes bound for camera parameters (pitch, yaw, distance, fov),
    opacity parameters (x, y, bandwidth), and color parameters (hues, saturations, lightnesses). It also provides
    utility methods to retrieve these bound.

    Attributes:
        config (dict): The configuration loaded from the provided config file.
        module (Module): An instance of the Module class, initialized with render configuration.
        camera_bound (SimpleNamespace): Bounds for camera parameters (pitch, yaw, distance, fov).
        opacity_bound (SimpleNamespace): Bounds for opacity parameters (x, y, bandwidth).
        color_bound (SimpleNamespace): Bounds for color parameters (hues, saturations, lightnesses).
        center (list): A list representing the center point [x, y, z], initialized to [0.0, 0.0, 0.0].

    Methods:
        __init__(self, config_file=None, volPath=None):
            Initializes the Bound class by reading the configuration file and setting up bound.

        get_bound(self, config, key, default_limit, default_data):
            Retrieves bound for a specific parameter from the configuration.

        _get_cam_bound(self, config):
            Initializes and returns bound for camera parameters.

        _get_opacity_bound(self, config):
            Initializes and returns bound for opacity parameters.

        _get_color_bound(self, config):
            Initializes and returns bound for color parameters.

        read_config(config_file):
            Static method to read and parse a JSON configuration file.
    """
    def __init__(self, config_file=None, volPath=None):
        self.config_file = config_file
        self.config = Bound.read_config(config_file)
        if self.config["color"]["type"] != "hsl":
            raise ValueError("Color type is not supported")

        # self.module=Module()
        # self.module.receive_refConfig(self.config["render_config"], volPath=self.config["vol_path"])
        self.gmm_bound=[self.config.get("min_op", self.config["max_op"]), self.config["max_op"]]
        self.camera_bound = self._get_cam_bound(self.config["camera"])
        self.opacity_bound = self._get_opacity_bound(self.config["opacity"])
        self.color_bound = self._get_color_bound(self.config["color"]["hsl"])
        # if there exists a config["camera"]["center"], use it
        if "center" in self.config["camera"]:
            self.center = self.config["camera"]["center"]
        else:
            self.center=[0.0,0.0,0.0]
        # get tf_size
        self.tf_size = self.config.get("tf_size", self.opacity_bound.x[1] + 1)
        
    def get_bound(self, config, key, default_limit, default_data):
        if config[key]["type"] == "random":
            return config.get(key, {}).get("limit", default_limit)
        else:
            return [config.get(key, {}).get("data", default_data), config.get(key, {}).get("data", default_data)]
    
    def _get_cam_bound(self, config):
        pitch_bound = self.get_bound(config, "pitch", [-np.pi / 2.0, np.pi / 2.0], [0.0, 0.0])
        yaw_bound = self.get_bound(config, "yaw", [0.0, np.pi * 2.0], [np.pi, np.pi])
        distance_bound = self.get_bound(config, "distance", [0.0, 1000], [0.0, 1.0])
        fov_bound = self.get_bound(config, "fov", [45.0, 60.0], [45.0, 45.0])
        camera_bound = SimpleNamespace(
            pitch = pitch_bound,
            yaw = yaw_bound,
            distance = distance_bound,
            fov = fov_bound,
        )
        
        return camera_bound
    
    def _get_opacity_bound(self, config):
        x_bound = self.get_bound(config, "x", [0.0, 255.0], [128.0, 128.0])
        y_bound = self.get_bound(config, "y", [0.0, 1.0], [0.5, 0.5])
        bandwidth_bound = self.get_bound(config, "bandwidth", [1.0, 15.0], [8.0, 8.0])
        opacity_bound = SimpleNamespace(
            x = x_bound,
            y = y_bound,
            bandwidth = bandwidth_bound,
        )
        
        return opacity_bound
    
    def _get_color_bound(self, config):
        hues_bound = self.get_bound(config, "hues", [0.0, 1.0], [0.5, 0.5])
        saturations_bound = self.get_bound(config, "saturations", [0.0, 1.0], [0.9, 0.9])
        lightnesses_bound = self.get_bound(config, "lightnesses", [0.0, 1.0], [0.5, 0.5])
        color_bound = SimpleNamespace(
            hues = hues_bound,
            saturations = saturations_bound,
            lightnesses = lightnesses_bound,
        )
        
        return color_bound
    
    @staticmethod
    def read_config(config_file):
        with open(config_file, 'r') as file_obj:
            config = json.load(file_obj)
        return config
        
    