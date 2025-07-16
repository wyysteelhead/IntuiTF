import torch
import numpy as np
from PIL import Image
from diffdvr.parametrizations import CameraOnASphere
from diffdvr.settings import setup_default_settings, setup_settings_from_settings
from diffdvr.utils import fibonacci_sphere
from diffdvr import renderer_dtype_torch, renderer_dtype_np
from genetic_optimize.TFparamsBase import TFparamsBase
from genetic_optimize.states.bound import Bound
from genetic_optimize.utils.image_utils import add_red_border, image_to_base64_pil
import pyrenderer

class TFparamsImp(TFparamsBase):
    global_inputs = None  # 类变量，在子类中设置具体值
    
    def __init__(self, id: int, bound: Bound = None, volume: pyrenderer.Volume = None, 
                 gradient=None, step_size=None, initial_rating=1600, W=512, H=512, 
                 bg_color=None, device="cuda", renderer_dtype_np = np.float64, tfparams=None, setInputs=False):
        """
        初始化 TFparamsImp 类
        
        Args:
            id: 唯一标识符
            bound: 参数边界约束
            volume: 体数据
            gradient: 梯度数据
            step_size: 步长
            initial_rating: 初始评分
            W, H: 图像宽高
            bg_color: 背景颜色
            device: 计算设备
            tfparams: 已有的传递函数参数对象
            setInputs: 是否设置渲染输入
        """
        assert (bound and volume) or (tfparams), "tfparams must be initialized with bound & volume or another instance"
        # 调用父类初始化
        super().__init__(id, bound, initial_rating, W, H, bg_color, device, renderer_dtype_np, tfparams)
        
        # 初始化渲染相关的输入设置
        if setInputs and bound is not None:
            self.__initialize_render_inputs(volume, gradient, step_size)
        elif setInputs and tfparams is not None:
            self.__initialize_settings(setInputs=setInputs)
    
    def __initialize_render_inputs(self, volume, gradient, step_size):
        """初始化渲染相关的输入设置"""
        if TFparamsImp.global_inputs is None:
            TFparamsImp.global_inputs = setup_default_settings(
                volume, gradient, TFparamsImp.W, TFparamsImp.H,
                step_size, True, True)
        self.inputs = setup_settings_from_settings(settings=TFparamsImp.global_inputs)
        
    def __initialize_settings(self, setInputs=True):
        if setInputs:
            self.inputs = setup_settings_from_settings(settings=TFparamsImp.global_inputs)
    
    def _get_copy(self):
        """重写父类的抽象方法"""
        copy_of_self = TFparamsImp(id=self.id, tfparams=self, setInputs=True)
        return copy_of_self
    
    def get_camera_matrix(self, camera=None):
        """获取相机矩阵"""
        if camera == None:
            camera = self.camera
        camera_module = CameraOnASphere(pyrenderer.Orientation(int(self.orientation)))
        pitch, yaw, distance, center, fov = camera
        cameras = camera_module(
            torch.from_numpy(np.array([center], dtype=renderer_dtype_np)).to(device=self.device),
            torch.from_numpy(np.array([yaw], dtype=renderer_dtype_np)).to(device=self.device).unsqueeze(1),
            torch.from_numpy(np.array([pitch], dtype=renderer_dtype_np)).to(device=self.device).unsqueeze(1),
            torch.from_numpy(np.array([distance], dtype=renderer_dtype_np)).to(device=self.device).unsqueeze(1))
        return cameras, fov
    
    def render_image_on_sphere(self, num_views=300, device="cuda", bg_color=None):
        """在球形表面上均匀分布相机位置，并渲染多个视角的图像"""
        if bg_color is None:
            bg_color = TFparamsBase.bg_color
            
        # 保存原始相机参数
        pitch, yaw, distance, center, fov = self.camera
        camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(num_views, dtype=renderer_dtype_np)
        camera_distance_cpu = distance * np.ones((num_views,), dtype=renderer_dtype_np)
        camera_center_cpu = np.stack([center] * num_views, axis=0).astype(dtype=renderer_dtype_np)
        camera_module = CameraOnASphere(self.orientation)
        cameras = camera_module(
            torch.from_numpy(camera_center_cpu).to(device=device),
            torch.from_numpy(camera_yaw_cpu).to(device=device).unsqueeze(1),
            torch.from_numpy(camera_pitch_cpu).to(device=device).unsqueeze(1),
            torch.from_numpy(camera_distance_cpu).to(device=device).unsqueeze(1))
        inputs = setup_settings_from_settings(settings=TFparamsImp.global_inputs)
        tf = self.get_tf()
        inputs.camera_mode = pyrenderer.CameraMode.ReferenceFrame
        inputs.camera = pyrenderer.CameraReferenceFrame(cameras, fov)
        inputs.tf_mode = pyrenderer.TFMode.Texture
        inputs.tf = tf
        output_color = torch.empty(num_views, TFparamsBase.H, TFparamsBase.W, 4, dtype=renderer_dtype_torch, device=device)
        output_termination_index = torch.empty(1, TFparamsBase.H, TFparamsBase.W, dtype=torch.int32, device=device)
        outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
        pyrenderer.Renderer.render_forward(inputs, outputs)
        img = output_color.cpu().numpy()[0]
        
        # 检查值范围并转换
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.clip(0, 1)
            img = (img * 255).astype(np.uint8)
        
        images = []
        for i in range(num_views):
            rgba_img = Image.fromarray(img[i])
            bg_img = Image.new('RGB', (TFparamsBase.W, TFparamsBase.H), bg_color)
            bg_img.paste(rgba_img, (0, 0), rgba_img)
            img_str = image_to_base64_pil(bg_img)
            images.append(img_str)
        return images
        
    def render_image(self, device="cuda", bg_color=None, force_render=False, setImage=True, addRedBorder=False, tf=None, transparent=False):
        """渲染单个图像"""
        if self.image and force_render == False:
            return self.image

        if bg_color is None:
            bg_color = TFparamsBase.bg_color
        camera_matrix, fov = self.get_camera_matrix()
        if tf is None:
            tf = self.get_tf()
        self.inputs.camera_mode = pyrenderer.CameraMode.ReferenceFrame
        self.inputs.camera = pyrenderer.CameraReferenceFrame(camera_matrix, fov)
        self.inputs.tf_mode = pyrenderer.TFMode.Texture
        self.inputs.tf = tf

        output_color = torch.empty(1, TFparamsBase.H, TFparamsBase.W, 4, dtype=renderer_dtype_torch, device=device)
        output_termination_index = torch.empty(1, TFparamsBase.H, TFparamsBase.W, dtype=torch.int32, device=device)
        outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
        pyrenderer.Renderer.render_forward(self.inputs, outputs)
        img = output_color.cpu().numpy()[0]
        
        # 检查值范围并转换
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.clip(0, 1)
            img = (img * 255).astype(np.uint8)

        # 转换为PIL图像
        rgba_img = Image.fromarray(img)
        
        if transparent is False:
            bg_img = Image.new('RGB', (TFparamsBase.W, TFparamsBase.H), bg_color)
            bg_img.paste(rgba_img, (0, 0), rgba_img)
        else:
            bg_img = rgba_img
        
        if setImage:
            self.image = bg_img
            self.img_str = image_to_base64_pil(bg_img)
        if addRedBorder:
            bg_img = add_red_border(bg_img)
            
        return bg_img