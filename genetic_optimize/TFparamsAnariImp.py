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
    使用 Anari (pynari) 渲染器实现的传递函数参数类
    """
    
    global_inputs = None  # 类变量，存储全局渲染设置
    
    def __init__(self, id: int, bound: Bound=None, volume: np.array = None, gradient: np.array = None, step_size = None, initial_rating=1600, W=512, H=512, bg_color=None, device="cuda", renderer_dtype_np = np.float64, tfparams=None, setInputs=False):
        super().__init__(id, bound, initial_rating, W, H, bg_color, device, renderer_dtype_np, tfparams)
        
        # 初始化渲染器设置
        if setInputs and bound is not None:
            self.__initialize_render_inputs(volume, gradient, step_size)
        elif setInputs and tfparams is not None:
            self.__initialize_settings(setInputs=setInputs)
            
    def __spherical_to_cartesian(self, pitch, yaw, distance, center=(0,0,0)):
        """球坐标转笛卡尔坐标"""
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.sin(pitch)
        z = distance * math.cos(pitch) * math.cos(yaw)
        return (center[0] + x, center[1] + y, center[2] + z)
            
    def __initialize_render_inputs(self, volume, gradient, step_size):
        """初始化渲染器的输入参数"""
        if volume is None:
            raise ValueError("必须提供体积数据")
            
        # 如果全局设置不存在，创建新的设置
        if TFparamsAnariImp.global_inputs is None:
            TFparamsAnariImp.global_inputs = setup_default_settings(
                volume_data=volume,
                screen_width=self.W,
                screen_height=self.H
            )
        
        # 从全局设置创建实例设置
        self.inputs = setup_settings_from_settings(TFparamsAnariImp.global_inputs)
        
    def __initialize_settings(self, setInputs=True):
        """从全局设置初始化实例设置"""
        if setInputs and TFparamsAnariImp.global_inputs is not None:
            self.inputs = setup_settings_from_settings(TFparamsAnariImp.global_inputs)
            
    def _get_copy(self):
        """重写父类的抽象方法"""
        copy_of_self = TFparamsAnariImp(id=self.id, tfparams=self, setInputs=True)
        return copy_of_self
        
    def render_image(self, device="cuda", bg_color=None, force_render=False, 
                    setImage=True, addRedBorder=False, tf=None, transparent=False):
        """渲染单个视角的图像"""
        if not hasattr(self, 'inputs') or not hasattr(self.inputs, 'device'):
            raise RuntimeError("渲染器未初始化，请先调用__initialize_render_inputs")
        
        if self.image is not None and not force_render:
            return self.image

        # 设置相机配置
        camera_config = {
            'pitch': 3.5,  # 约90度仰角
            'yaw': 2.2,    # 0度偏航角
            'distance': 10.0
        }

        # 如果没有提供传输函数，使用当前的传输函数
        if tf is None:
            tf = self.get_tf().cpu().numpy()[0]  # 移除batch维度
        if bg_color is None:
            bg_color = self.bg_color

        # 使用renderer.py中的render_volume函数
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
        """在球面上渲染多个视角的图像"""
        if not hasattr(self, 'inputs') or not hasattr(self.inputs, 'device'):
            raise RuntimeError("渲染器未初始化，请先调用__initialize_render_inputs")
            
        images = []
        radius = 3.4  # 相机距离
        
        # 生成球面上的均匀分布点
        phi = np.pi * (3 - np.sqrt(5))  # 黄金角
        for i in range(num_views):
            y = 1 - (i / float(num_views - 1)) * 2  # y从1到-1
            radius_at_y = np.sqrt(1 - y * y)  # 当前y位置的圆半径
            
            theta = phi * i  # 方位角
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            # 计算球面坐标
            pitch = np.arcsin(y)
            yaw = np.arctan2(x, z)
            
            # 设置当前视角的相机配置
            camera_config = {
                'pitch': pitch,
                'yaw': yaw,
                'distance': radius
            }

            # 使用renderer.py中的render_volume函数
            im = render_volume(
                volume_data=self.inputs.volume_data,
                volume_dims=self.inputs.volume_dims,
                W=self.W,
                H=self.H,
                camera_config=camera_config,
                tf=self.get_tf().cpu().numpy()[0],  # 使用当前的传输函数
                pixel_samples=1024 if pynari.has_cuda_capable_gpu() else 16
            )
            
            images.append(im)
            
        return images