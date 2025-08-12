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
    global_inputs = None  # Class variable, set specific values in subclasses
    
    def __init__(self, id: int, bound: Bound = None, volume: pyrenderer.Volume = None, 
                 gradient=None, step_size=None, initial_rating=1600, W=512, H=512, 
                 bg_color=None, device="cuda", renderer_dtype_np = np.float64, tfparams=None, setInputs=False):
        """
        Initialize TFparamsImp class
        
        Args:
            id: Unique identifier
            bound: Parameter boundary constraints
            volume: Volume data
            gradient: Gradient data
            step_size: Step size
            initial_rating: Initial rating
            W, H: Image width and height
            bg_color: Background color
            device: Computing device
            tfparams: Existing transfer function parameter object
            setInputs: Whether to set rendering inputs
        """
        assert (bound and volume) or (tfparams), "tfparams must be initialized with bound & volume or another instance"
        # Call parent class initialization
        super().__init__(id, bound, initial_rating, W, H, bg_color, device, renderer_dtype_np, tfparams)
        
        # Initialize rendering-related input settings
        if setInputs and bound is not None:
            self.__initialize_render_inputs(volume, gradient, step_size)
        elif setInputs and tfparams is not None:
            self.__initialize_settings(setInputs=setInputs)
    
    def __initialize_render_inputs(self, volume, gradient, step_size):
        """Initialize rendering-related input settings"""
        if TFparamsImp.global_inputs is None:
            TFparamsImp.global_inputs = setup_default_settings(
                volume, gradient, TFparamsImp.W, TFparamsImp.H,
                step_size, True, True)
        self.inputs = setup_settings_from_settings(settings=TFparamsImp.global_inputs)
        
    def __initialize_settings(self, setInputs=True):
        if setInputs:
            self.inputs = setup_settings_from_settings(settings=TFparamsImp.global_inputs)
    
    def _get_copy(self):
        """Override parent class abstract method"""
        copy_of_self = TFparamsImp(id=self.id, tfparams=self, setInputs=True)
        return copy_of_self

    def load_render_settings(self, bound, volume, gradient, step_size, bg_color=(255,255,255), W=512, H=512):
        super().load_render_settings(bound=bound, bg_color=bg_color, W=W, H=H)
        if self.global_inputs is None:
            self.global_inputs = setup_default_settings(
            volume, gradient, self.W, self.H,
            step_size, True, True)
        self.inputs = setup_settings_from_settings(settings=self.global_inputs)
    
    def get_camera_matrix(self, camera=None):
        """Get camera matrix"""
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
        """Uniformly distribute camera positions on spherical surface and render multi-view images"""
        if bg_color is None:
            bg_color = TFparamsBase.bg_color
            
        # Save original camera parameters
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
        
        # Check value range and convert
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
        """Render single image"""
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
        
        # Check value range and convert
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.clip(0, 1)
            img = (img * 255).astype(np.uint8)

        # Convert to PIL image
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