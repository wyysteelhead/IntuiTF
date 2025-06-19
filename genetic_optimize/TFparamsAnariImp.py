import numpy as np
import torch
from PIL import Image
from genetic_optimize.TFparamsBase import TFparamsBase
from genetic_optimize.states.bound import Bound
from genetic_optimize.utils.image_utils import image_to_base64_pil, add_red_border
import pynari as anari
import tempfile
import io


class TFparamsAnariImp(TFparamsBase):
    """
    使用 Anari (pynari) 渲染器实现的传递函数参数类
    """
    
    def __init__(self, id: int, bound: Bound=None, volume: np.array = None, gradient: np.array = None, step_size = None, initial_rating=1600, W=512, H=512, bg_color=None, device="cuda", tfparams=None, setRenderer=False):
        super().__init__(id, bound, initial_rating, W, H, bg_color, device, tfparams)
        
        # 初始化 Anari 设备和世界
        if setRenderer:
            self.__initialize_render_inputs(volume, gradient, step_size)
        
    #TODO
    def __initialize_render_inputs(self, volume, gradient, step_size):
        pass
            
    def __get_copy(self):
        """创建当前对象的深拷贝"""
        return TFparamsAnariImp(
            id=self.id,
            tfparams=self
        )
     
    #TODO   
    def get_tf(self, color=None, opacity=None):
        """获取传递函数张量"""
        if color is None:
            color = self.color
        if opacity is None:
            opacity = self.opacity
            
        # 创建传递函数张量
        tf_size = getattr(self, 'tf_size', 256)
        
        # 将高斯参数转换为传递函数
        tf_values = np.zeros((tf_size, 4), dtype=np.float32)  # RGBA
        
        x_coords = np.linspace(0, 1, tf_size)
        
        for i, x in enumerate(x_coords):
            rgba = [0, 0, 0, 0]
            
            # 对每个高斯进行采样
            for j in range(len(self.gaussians)):
                gaussian = self.gaussians[j]
                center = gaussian.opacity[0]
                bandwidth = gaussian.opacity[1]
                height = gaussian.opacity[2]
                
                # 计算高斯值
                gauss_val = height * np.exp(-0.5 * ((x - center) / bandwidth) ** 2)
                
                # 累加颜色和不透明度
                rgba[0] += gauss_val * gaussian.color[0]
                rgba[1] += gauss_val * gaussian.color[1]
                rgba[2] += gauss_val * gaussian.color[2]
                rgba[3] += gauss_val
                
            tf_values[i] = rgba
            
        # 转换为 PyTorch 张量
        tf_tensor = torch.from_numpy(tf_values).unsqueeze(0)  # 添加batch维度
        
        return tf_tensor
    
    #TODO 
    def render_image(self, device="cuda", bg_color=None, force_render=False, 
                    setImage=True, addRedBorder=False, tf=None, transparent=False):
        pass
    
    #TODO
    def render_image_on_sphere(self, num_views=300, device="cuda", bg_color=None):
        pass