from abc import abstractmethod
import colorsys
import io
import random
import tempfile
import cv2
from matplotlib import pyplot as plt
import numpy as np
# import torch
from genetic_optimize.states.bound import Bound
from genetic_optimize.utils.dtype_utils import get_renderer_dtypes
# from genetic_optimize.states.direction import Direction
from genetic_optimize.states.gaussian import Gaussian
from genetic_optimize.states.genetic_config import GeneticConfig
from genetic_optimize.utils.image_utils import add_red_border, apply_semi_transparent_background, image_to_base64_pil
from genetic_optimize.utils.thread import ParallelExecutor
# import pyrenderer
from PIL import Image
import uuid
import torch
from PIL import ImageFilter, ImageDraw, ImageEnhance

class TFparamsBase:    
    W = 512
    H = 512
    max_opacity=1.0
    min_opacity=0.0
    bg_color = (0,0,0)
    renderer_dtype_np = np.float64

    def __init__(self, id: int, bound : Bound = None, initial_rating=1600, W = 512, H = 512, bg_color = None, device="cuda", renderer_dtype_np = np.float64, tfparams = None):
        self.image = None
        self.img_str = None
        self.initial_rating=initial_rating
        self.parent_id = None
        self.inputs = None
        self.renderer_dtype_np = renderer_dtype_np
        if bg_color is not None:
            TFparamsBase.bg_color = bg_color
        if bound:
            self.__initialize_from_bound(id, bound, W, H, device)
        else:
            self.__initialize_from_tfparams(tfparams)
            
    def __initialize_from_bound(self, id, bound: Bound, W, H, device):
        self.gmm_num = int(random.uniform(bound.gmm_bound[0], bound.gmm_bound[1]))
        self.camera = self.__gen_camera(bound)
        self.opacity = self.__gen_opacity(bound)
        self.color = self.__gen_color(bound, self.gmm_num)
        self.gaussians = Gaussian.setup_gaussians(self.opacity, self.color)
        TFparamsBase.H = H
        TFparamsBase.W = W
        self.tf_size = bound.tf_size
        self.orientation = bound.config['camera']['orientation']
        self.device = device
        self.id = id
        self.rating = self.initial_rating
        self.matches = set()  # 记录对战过的对手ID
        TFparamsBase.max_opacity = max(TFparamsBase.max_opacity, bound.opacity_bound.y[1])
        TFparamsBase.min_opacity = min(TFparamsBase.min_opacity, bound.opacity_bound.y[0])
            
    def __initialize_from_tfparams(self, tfparams):
        """根据已有的 tfparams 初始化，创建独立的数据副本"""
        self.gmm_num = tfparams.gmm_num
        self.camera = tuple(tfparams.camera)  # Create a new tuple
        self.opacity = np.copy(tfparams.opacity)  # Create a deep copy of numpy arrays
        self.color = np.copy(tfparams.color)
        self.gaussians = [tfparams.gaussians[i].copy() for i in range(len(tfparams.gaussians))]
        self.tf_size = tfparams.tf_size
        self.orientation = tfparams.orientation
        self.device = tfparams.device
        self.id = tfparams.id
        self.rating = tfparams.rating
        self.matches = set(tfparams.matches)  # Create a new set with the same values
        self.renderer_dtype_np = tfparams.renderer_dtype_np
    
    def to_json(self):
        """
        将 TFparams 对象转换为 JSON 可序列化的字典。
        """
        return {
            'id': self.id,
            'rating': self.rating,
            'camera': self.camera,
            'gaussians': [gaussian.to_json() for gaussian in self.gaussians],
            "image": self.img_str
        }
            
    def set_bg_color(self, color):
        TFparamsBase.bg_color = color
            
    def reset_matching(self, id, bound=None):
        self.id = id
        self.rating = self.initial_rating
        self.matches = set()
        self.image = None
        self.img_str = None
        if bound:
            self.camera = self.__gen_camera(bound)
        for i in range(len(self.gaussians)):
            self.gaussians[i].id = i
    
    @abstractmethod
    def _get_copy(self):
        pass
        
    def set_from(self, tfparamsbase):
        self.id = tfparamsbase.id
        self.gmm_num = tfparamsbase.gmm_num
        self.tf_size = tfparamsbase.tf_size
        self.orientation = tfparamsbase.orientation
        self.device = tfparamsbase.device
        self.image = None if tfparamsbase.image is None else tfparamsbase.image.copy()  # Create a copy of the image if it exists
        self.img_str = None if tfparamsbase.img_str is None else tfparamsbase.img_str  # Create a copy of the image string if it exists
        self.rating = tfparamsbase.rating
        self.matches = tfparamsbase.matches  # 记录对战过的对手ID
        
    def random_color(self, bound: Bound):
        self.image = None
        self.color = self.__gen_color(bound, self.gmm_num)
        for i in range(len(self.gaussians)):
            if self.gaussians[i].is_frozen():
                continue
            # copy the color instead of reference
            self.gaussians[i].color = np.copy(self.color[i])
            
    def random_opacity(self, bound: Bound):
        self.image = None
        self.opacity = self.__gen_opacity(bound, self.gmm_num)
        for i in range(len(self.gaussians)):
            if self.gaussians[i].is_frozen():
                continue
            # copy the opacity instead of reference
            self.gaussians[i].opacity[2] = np.copy(self.opacity[i][2])
            self.gaussians[i].opacity[1] = np.copy(self.opacity[i][1])
            
    def update_camera_rotation(self, direction):
        """
        根据鼠标/触控拖拽更新相机旋转角度
        
        Args:
            direction (dict): 包含 deltaX 和 deltaY 的字典，表示拖拽偏移量
        """
        # 提取拖拽偏移量
        delta_x = direction.get('deltaX', 0)
        delta_y = direction.get('deltaY', 0)
        
        # 定义旋转灵敏度系数（可以根据需求调整）
        sensitivity = 0.5
        
        # 当前相机参数
        pitch, yaw, distance, center, fov = self.camera
        
        # 根据水平拖拽更新偏航角（yaw）- 水平拖拽影响左右旋转
        # 负号是因为向右拖动（正deltaX）应该减少yaw（相机向左转）
        yaw = (yaw - delta_x * sensitivity) % 360
        
        # 根据垂直拖拽更新俯仰角（pitch）- 垂直拖拽影响上下俯仰
        # 负号是因为向下拖动（正deltaY）应该减少pitch（相机向上看）
        pitch += delta_y * sensitivity
        
        # 限制pitch范围，防止过度旋转（通常在-90到90度之间）
        pitch = max(-85, min(85, pitch))
        
        # 更新相机参数
        self.camera = (pitch, yaw, distance, center, fov)
        self.image = None
        self.img_str = None

    def update_camera_zoom(self, zoom_factor=0.1):
        """
        根据缩放因子更新相机的缩放级别
        
        Args:
            zoom_factor (float): 缩放因子，正值表示放大，负值表示缩小
        """
        # 当前相机参数
        pitch, yaw, distance, center, fov = self.camera
        # 更新距离
        distance *= (1 - zoom_factor)
        # 更新相机参数
        self.camera = (pitch, yaw, distance, center, fov)
        self.image = None
        self.img_str = None
        
    def __gen_camera(self, bound : Bound):
        pitch = random.uniform(bound.camera_bound.pitch[0], bound.camera_bound.pitch[1])
        yaw = random.uniform(bound.camera_bound.yaw[0], bound.camera_bound.yaw[1])
        distance = random.uniform(bound.camera_bound.distance[0], bound.camera_bound.distance[1])
        fov = random.uniform(bound.camera_bound.fov[0], bound.camera_bound.fov[1])
        center = bound.center
        return pitch, yaw, distance, center, fov
    
    def __gen_opacity(self, bound : Bound, num=None):
        if num == None: 
            num = self.gmm_num
        x = np.random.uniform(bound.opacity_bound.x[0], bound.opacity_bound.x[1], size=(num, 1)).astype(self.renderer_dtype_np)
        y = np.random.uniform(bound.opacity_bound.y[0], bound.opacity_bound.y[1], size=(num, 1)).astype(self.renderer_dtype_np)
        bandwidth = np.random.uniform(bound.opacity_bound.bandwidth[0], bound.opacity_bound.bandwidth[1], size=(num, 1)).astype(self.renderer_dtype_np)
        opacity = np.concatenate((x, bandwidth, y), axis=1)
        # Sort according to x values (first column), smallest x at the front
        return opacity[opacity[:, 0].argsort()]
    
    def __gen_hsl(self, bound : Bound, num):
        hues = np.random.uniform(low=bound.color_bound.hues[0], high=bound.color_bound.hues[1], size=(num,)).astype(self.renderer_dtype_np)
        saturations = np.random.uniform(low=bound.color_bound.saturations[0], high=bound.color_bound.saturations[1], size=(num,)).astype(self.renderer_dtype_np)
        lightnesses = np.random.uniform(low=bound.color_bound.lightnesses[0], high=bound.color_bound.lightnesses[1], size=(num,)).astype(self.renderer_dtype_np)
        colors_hsl = np.column_stack((hues, saturations, lightnesses))
        colors_rgb = np.asarray(np.apply_along_axis(lambda x: np.array(colorsys.hls_to_rgb(x[0], x[2], x[1]), dtype=self.renderer_dtype_np), 1, colors_hsl))
        color = np.random.permutation(colors_rgb)
        return color

    def __gen_color(self, bound: Bound, num=None):
        if num == None: 
            num = self.gmm_num
        #对于每个高斯峰坐标，随机颜色
        random_color = self.__gen_hsl(bound, num)
        return random_color
    
    def freeze_all_gaussian(self):
        # self.selected_gaussians = [0 for i in range(self.gmm_num)]
        for i in range(len(self.gaussians)):
            self.gaussians[i].freeze()
        
    def add_random_gaussian(self, bound, num=1):
        self.gmm_num += num
        opacity = self.__gen_opacity(bound, num)
        color = self.__gen_color(bound, num)
        new_gaussians = Gaussian.setup_gaussians(opacity, color)
        self.gaussians.extend(new_gaussians)
        # self.selected_gaussians.extend([i for i in range(self.gmm_num - num, self.gmm_num)])
            
    @staticmethod
    def crossover(TF1, TF2):
        """
        Perform crossover operation on two transfer functions (TF1 and TF2) to generate two offspring.
        The crossover operation involves three main components:
        1. Camera parameters
        2. Opacity parameters
        3. Color parameters
        Parameters:
        TF1 (object): The first transfer function object.
        TF2 (object): The second transfer function object.
        Returns:
        list: A list containing the two offspring transfer function objects [child1, child2].
        """
        # Create copies of the parents
        child1 = TF1._get_copy()
        child2 = TF2._get_copy()
        child1.parent_id = [TF1.id, TF2.id]
        child2.parent_id = [TF1.id, TF2.id]
        
        def __cross_camera(rate=0.5):
            # 提取 camera 参数
            pitch1, yaw1, distance1, center1, fov1 = TF1.camera
            pitch2, yaw2, distance2, center2, fov2 = TF2.camera

            # 创建两个子代，分别进行交叉
            child1_pitch = pitch1 if random.random() > rate else pitch2
            child1_yaw = yaw1 if random.random() > rate else yaw2
            child1_distance = distance1 if random.random() > rate else distance2
            child1_center = center1 if random.random() > rate else center2
            child1_fov = fov1 if random.random() > rate else fov2

            child2_pitch = pitch2 if random.random() > rate else pitch1
            child2_yaw = yaw2 if random.random() > rate else yaw1
            child2_distance = distance2 if random.random() > rate else distance1
            child2_center = center2 if random.random() > rate else center1
            child2_fov = fov2 if random.random() > rate else fov1
            
            child1.camera = (child1_pitch, child1_yaw, child1_distance, child1_center, child1_fov)
            child2.camera = (child2_pitch, child2_yaw, child2_distance, child2_center, child2_fov)
        
        def __cross_opacity(rate=0.5):
            child1_opacity = []
            child2_opacity = []

            # 对每个点的参数进行交叉操作
            for i in range(TF1.opacity.shape[0]):  # 遍历每个样本点
                child1_point = []
                child2_point = []
                # 按照 rate 决定是否整行交换
                if random.random() < rate:
                    # 整行交换
                    child1_point = list(TF2.opacity[i])
                    child2_point = list(TF1.opacity[i])
                else:
                    # 保持原样
                    child1_point = list(TF1.opacity[i])
                    child2_point = list(TF2.opacity[i])

                # 将交叉后的结果存入子代
                child1_opacity.append(child1_point)
                child2_opacity.append(child2_point)

            # 将交叉后的列表转换为 numpy 数组
            child1.opacity = np.array(child1_opacity)
            child2.opacity = np.array(child2_opacity)

            # Update x values for color points
            child1.__update_x()
            child2.__update_x()
        
        def __cross_color(rate=0.5):
            child1_color = []
            child2_color = []

            # 对每个点的参数进行交叉操作
            for i in range(TF1.color.shape[0]):  # 遍历每个样本点
                child1_point = [TF1.color[i, 0]]
                child2_point = [TF2.color[i, 0]]
                # For first and last points, keep the original color
                if i == 0 or i == TF1.color.shape[0] - 1:
                    for j in range(1, TF1.color.shape[1]):
                        child1_point.append(TF1.color[i, j])
                        child2_point.append(TF2.color[i, j])
                # For middle points, swap entire color based on rate
                elif random.random() < rate:
                    # Swap entire color (RGB values)
                    for j in range(1, TF1.color.shape[1]):
                        child1_point.append(TF2.color[i, j])
                        child2_point.append(TF1.color[i, j])
                else:
                    # Keep original colors
                    for j in range(1, TF1.color.shape[1]):
                        child1_point.append(TF1.color[i, j])
                        child2_point.append(TF2.color[i, j])

                # 将交叉后的结果存入子代
                child1_color.append(child1_point)
                child2_color.append(child2_point)

            # 将交叉后的列表转换为 numpy 数组
            child1.color = np.array(child1_color)
            child2.color = np.array(child2_color)
            
        def __cross_all_old(rate=0.5):
            # 对每个点的参数进行交叉操作
            i1 = 0
            i2 = 0
            while((i1 < TF1.gmm_num and i2 < TF2.gmm_num)):  # 遍历每个样本点
                while (i1 < TF1.gmm_num and TF1.gaussians[i1].is_frozen()):
                    i1 += 1
                while (i2 < TF2.gmm_num and TF2.gaussians[i2].is_frozen()):
                    i2 += 1
                
                # Check if indices are still within bound after skipping frozen gaussians
                if i1 >= TF1.gmm_num or i2 >= TF2.gmm_num:
                    break
                    
                # 按照 rate 决定是否整行交换
                if random.random() < rate:
                    # 整行交换
                    child1.gaussians[i1] = TF2.gaussians[i2].copy()
                    child2.gaussians[i2] = TF1.gaussians[i1].copy()
                i1 += 1
                i2 += 1
                
        def __cross_all(rate=0.6):
            """
            增强版的高斯参数交叉操作，支持属性级别的精细交叉
            
            Args:
                rate (float): 进行交叉的基础概率
            """
            # 对每个点的参数进行交叉操作
            i1 = 0
            i2 = 0
            while((i1 < TF1.gmm_num and i2 < TF2.gmm_num)):  # 遍历每个样本点
                while (i1 < TF1.gmm_num and TF1.gaussians[i1].is_frozen()):
                    i1 += 1
                while (i2 < TF2.gmm_num and TF2.gaussians[i2].is_frozen()):
                    i2 += 1
                
                # 检查索引是否仍在有效范围内
                if i1 >= TF1.gmm_num or i2 >= TF2.gmm_num:
                    break
                    
                # 针对每个参数单独决定是否交换
                cross_decision = {
                    'x': random.random() < rate,
                    'bandwidth': random.random() < rate,
                    'y': random.random() < rate,
                    'color': random.random() < rate
                }
                
                # 根据交叉决策执行交换操作
                if cross_decision['x']:
                    # 交换 x 坐标
                    temp_x = child1.gaussians[i1].opacity[0]
                    child1.gaussians[i1].opacity[0] = child2.gaussians[i2].opacity[0]
                    child2.gaussians[i2].opacity[0] = temp_x
                    
                if cross_decision['bandwidth']:
                    # 交换带宽
                    temp_bandwidth = child1.gaussians[i1].opacity[1]
                    child1.gaussians[i1].opacity[1] = child2.gaussians[i2].opacity[1]
                    child2.gaussians[i2].opacity[1] = temp_bandwidth
                    
                if cross_decision['y']:
                    # 交换不透明度高度
                    temp_y = child1.gaussians[i1].opacity[2]
                    child1.gaussians[i1].opacity[2] = child2.gaussians[i2].opacity[2]
                    child2.gaussians[i2].opacity[2] = temp_y
                    
                if cross_decision['color']:
                    # 交换整个颜色
                    temp_color = np.copy(child1.gaussians[i1].color)
                    child1.gaussians[i1].color = np.copy(child2.gaussians[i2].color)
                    child2.gaussians[i2].color = temp_color
                    
                # 如果没有任何参数进行交换，则有一定概率强制交换一个随机参数
                if not any(cross_decision.values()) and random.random() < 0.5:
                    # 随机选择一个参数进行交换
                    param = random.choice(['x', 'bandwidth', 'y', 'color'])
                    if param == 'x':
                        temp_x = child1.gaussians[i1].opacity[0]
                        child1.gaussians[i1].opacity[0] = child2.gaussians[i2].opacity[0]
                        child2.gaussians[i2].opacity[0] = temp_x
                    elif param == 'bandwidth':
                        temp_bandwidth = child1.gaussians[i1].opacity[1]
                        child1.gaussians[i1].opacity[1] = child2.gaussians[i2].opacity[1]
                        child2.gaussians[i2].opacity[1] = temp_bandwidth
                    elif param == 'y':
                        temp_y = child1.gaussians[i1].opacity[2]
                        child1.gaussians[i1].opacity[2] = child2.gaussians[i2].opacity[2]
                        child2.gaussians[i2].opacity[2] = temp_y
                    elif param == 'color':
                        temp_color = np.copy(child1.gaussians[i1].color)
                        child1.gaussians[i1].color = np.copy(child2.gaussians[i2].color)
                        child2.gaussians[i2].color = temp_color
                        
                i1 += 1
                i2 += 1
            
        # __cross_camera()
        # __cross_opacity()
        # __cross_color()
        __cross_all()
        
        # Reset the image of the children to None since parameters have changed
        child1.image = None
        child2.image = None
        
        return [child1, child2]
        
    def mutate(self, bound: Bound, genetic_config: GeneticConfig, iter=0, maxiter=10, directions = None):
        def mutate_param(rate_factor, scale_factor, bound, param, direction):
            if random.random() < rate_factor:
                dx = np.random.normal(0, scale_factor * (bound[1] - bound[0]))
                # print(dx)
                if direction is not None:
                    dx = abs(dx) * direction
                param += dx
                param = np.clip(param, bound[0], bound[1])
            return param
        
        # 1. Mutate camera parameters
        def __mutate_camera():
            # 计算变异尺度
            # 判断 mutation_scale 是否为单一数值或范围元组
            camera_bound = bound.camera_bound
            scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.cam_mutation_scale, iter=iter, maxiter=maxiter)
            rate_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.cam_mutation_rate, iter=iter, maxiter=maxiter)
            
            pitch, yaw, distance, center, fov = self.camera
            # Use mutate_param function to apply mutations to each camera parameter
            pitch = mutate_param(rate_factor, scale_factor, camera_bound.pitch, pitch, directions.camera.pitch if directions is not None and hasattr(directions.camera, 'pitch') else None)
            yaw = mutate_param(rate_factor, scale_factor, camera_bound.yaw, yaw, directions.camera.yaw if directions is not None and hasattr(directions.camera, 'yaw') else None)
            distance = mutate_param(rate_factor, scale_factor, camera_bound.distance, distance, directions.camera.distance if directions is not None and hasattr(directions.camera, 'distance') else None)
            # Center is not mutated as per original code
            fov = mutate_param(rate_factor, scale_factor, camera_bound.fov, fov, directions.camera.fov if directions is not None and hasattr(directions.camera, 'fov') else None)
            self.camera = (pitch, yaw, distance, center, fov)
        
        # 2. Mutate opacity parameters
        def __mutate_opacity():
            opacity_bound = bound.opacity_bound
            op_scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.op_mutation_scale, iter=iter, maxiter=maxiter)
            x_scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.x_mutation_scale, iter=iter, maxiter=maxiter)
            bandwidth_scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.bandwidth_mutation_scale, iter=iter, maxiter=maxiter)
            rate_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.op_mutation_rate, iter=iter, maxiter=maxiter)
            
            # Define threshold for "low opacity"
            # low_opacity_threshold = 0.1
            
            for i in range(len(self.gaussians)):
                if self.gaussians[i].is_frozen():
                    continue
                if genetic_config.text_mode == False:
                    x = mutate_param(rate_factor, x_scale_factor, opacity_bound.x, self.gaussians[i].opacity[0], directions.opacity.x if directions is not None and hasattr(directions.opacity, 'x') else None)
                else:
                    x = mutate_param(rate_factor * 0.1, x_scale_factor * 0.1, opacity_bound.x, self.gaussians[i].opacity[0], directions.opacity.x if directions is not None and hasattr(directions.opacity, 'x') else None)
                bandwidth = mutate_param(rate_factor, bandwidth_scale_factor, opacity_bound.bandwidth, self.gaussians[i].opacity[1], directions.opacity.bandwidth if directions is not None and hasattr(directions.opacity, 'bandwidth') else None)
                y = mutate_param(rate_factor, op_scale_factor, opacity_bound.y, self.gaussians[i].opacity[2], directions.opacity.y if directions is not None and hasattr(directions.opacity, 'y') else None)
                
                self.gaussians[i].opacity = np.array([x, bandwidth, y])
                
        # 3. Mutate color parameters
        def __mutate_color():
            color_bound = bound.color_bound
            H_scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.H_mutation_scale, iter=iter, maxiter=maxiter)
            SL_scale_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.SL_mutation_scale, iter=iter, maxiter=maxiter)
            rate_factor = GeneticConfig.get_mutation_factor(mutation_factor=genetic_config.color_mutation_rate, iter=iter, maxiter=maxiter)
            
            if random.random() > 0.7:
                # 随机交换两个颜色，而不是打乱所有颜色
                unfrozen_indices = [i for i in range(len(self.gaussians)) if not self.gaussians[i].is_frozen()]
                
                # 只有当有至少两个未冻结的高斯时才执行交换
                if len(unfrozen_indices) >= 2:
                    # 随机选择两个不同的索引
                    idx1, idx2 = random.sample(unfrozen_indices, 2)
                    
                    # 交换这两个高斯的颜色
                    temp_color = np.copy(self.gaussians[idx1].color)
                    self.gaussians[idx1].color = np.copy(self.gaussians[idx2].color)
                    self.gaussians[idx2].color = temp_color
            else:
            # 跳过第一个和最后一个颜色点
                for i in range(len(self.gaussians)):
                        if self.gaussians[i].is_frozen():
                            continue
                    # 只有当随机概率小于变异率时才进行变异
                        # 从RGB转换到HLS颜色空间
                        hues, lightnesses, saturations = colorsys.rgb_to_hls(self.gaussians[i].color[0], self.gaussians[i].color[1], self.gaussians[i].color[2])
                        
                        # 随机选择一个通道进行变异 (0=hue, 1=saturation, 2=lightness)
                        # channel = random.randint(0, 2)
                        # if channel == 0:  # 变异色相(hue)
                        if random.random() < rate_factor:
                            if directions is not None and directions.color.hues is not None:
                                hues = directions.color.hues
                            else:
                                # 感知敏感的色相变异改进（保持0-1范围）
                                hue_span = color_bound.hues[1] - color_bound.hues[0]
                                red_zone = 0.05  # 红色敏感区边界（对应0-0.05和0.95-1.0）
                                # 动态步长调整（红色区域步长缩减60%）
                                H_scale = np.where(((hues < red_zone) | (hues > 1-red_zone)), 
                                                H_scale_factor * 0.4,  # 红色区
                                                H_scale_factor) * hue_span  # 常规区
                                # 环形空间平滑变异（通过极坐标避免突变）
                                theta = hues * 2 * np.pi  # 映射到极角
                                dh = np.random.normal(0, H_scale)  # 生成变异量
                                hues = ((theta + dh) % (2*np.pi)) / (2*np.pi)  # 回环映射
                                # 约束到目标区间
                                hues = hues * hue_span + color_bound.hues[0]
                                hues = np.clip(hues, color_bound.hues[0], color_bound.hues[1])
                        # elif channel == 1:  # Saturation mutation
                        saturations = mutate_param(rate_factor, SL_scale_factor, color_bound.saturations, 
                                                        saturations, 
                                                        directions.color.saturations if directions is not None and hasattr(directions.color, 'saturations') else None)
                        # else:  # Lightness mutation
                        lightnesses = mutate_param(rate_factor, SL_scale_factor, color_bound.lightnesses, 
                                                        lightnesses,
                                                        directions.color.lightnesses if directions is not None and hasattr(directions.color, 'lightnesses') else None)
                        
                        # 转换回RGB
                        r, g, b = colorsys.hls_to_rgb(hues, lightnesses, saturations)
                        self.gaussians[i].color[0] = np.clip(r, 0, 1)
                        self.gaussians[i].color[1] = np.clip(g, 0, 1)
                        self.gaussians[i].color[2] = np.clip(b, 0, 1)
                    
        __mutate_camera()
        __mutate_opacity()
        __mutate_color()
    
    @abstractmethod
    def render_image_on_sphere(self, num_views=300, device="cuda", bg_color=None):
        pass
        
    def get_tf(self, color=None, opacity=None):
        """生成传递函数张量"""
        # 按不透明度x坐标排序高斯点
        self.gaussians = sorted(self.gaussians, key=lambda x: x.opacity[0])
        
        if opacity is None:
            for i in range(len(self.gaussians)):
                # 堆叠不透明度
                if i == 0:
                    opacity = np.array([self.gaussians[i].opacity])
                else:
                    opacity = np.vstack([opacity, self.gaussians[i].opacity])
                    
        if color is None:
            for i in range(len(self.gaussians)):
                # 堆叠颜色
                if i == 0:
                    color = np.array([self.gaussians[i].color])
                else:
                    color = np.vstack([color, self.gaussians[i].color])
                    
        color = np.concatenate((opacity[:, 0:1], color), axis=1)
                    
        c_left = np.array([[0, self.gaussians[0].color[0], self.gaussians[0].color[1], self.gaussians[0].color[2]]], dtype=self.renderer_dtype_np)
        c_right = np.array([[self.tf_size, self.gaussians[-1].color[0], self.gaussians[-1].color[1], self.gaussians[-1].color[2]]], dtype=self.renderer_dtype_np)
        color = np.vstack([c_left, color, c_right])
        opacity, color = self.__update_x(opacity, color)
        
        color_tf = self.__gen_ctf_from_gmm(color, 0, 255)
        opacity_tf = self.__gen_otf_from_gmm(opacity, 0, 255)
        ctf = color_tf[:, 1:]
        otf = opacity_tf[:, 1:]
        tf = np.concatenate([ctf, otf], axis=1)
        tf = torch.from_numpy(np.array([tf], dtype=self.renderer_dtype_np)).to(device=self.device)
        return tf
        
    @abstractmethod
    def render_image(self, device="cuda", bg_color=None, force_render=False, setImage=True, addRedBorder=False, tf=None, transparent = False):
        pass
    
    def check_gaussian_id(self):
        # check if the self.gaussians[].id is duplicate
        ids = [gaussian.id for gaussian in self.gaussians]
        if len(ids) != len(set(ids)):
            for i in range(len(self.gaussians)):
                self.gaussians[i].id = i
    
    def render_seperate_gaussians(self, device="cuda", bg_color=None):
        # copy self.gaussians
        gaussians = []
        for i in range (len(self.gaussians)):
            gaussians.append(self.gaussians[i].copy())
        
        for i in range(len(gaussians)):
            # set y to zero
            gaussians[i].opacity[2] = 0
        
        results = []
        
        for index in range(len(gaussians)):
            id = gaussians[index].id
            gaussians[index].opacity[2] = self.gaussians[index].opacity[2]
            for i in range(len(gaussians)):
                # stack opacity
                if i == 0:
                    opacity = np.array([gaussians[i].opacity])
                else:
                    opacity = np.vstack([opacity, gaussians[i].opacity])
            for i in range(len(gaussians)):
                # stack color
                if i == 0:
                    color = np.array([gaussians[i].color])
                else:
                    color = np.vstack([color, gaussians[i].color])
                    
            tf = self.get_tf(color, opacity)
            image = self.render_image(device, bg_color, force_render=True, setImage=False, addRedBorder=False, tf=tf, transparent=True)
            
            back_image = self.render_image(device)
            # Convert background to grayscale
            back_image_gray = back_image.convert('L').convert('RGB')
            
            back_image_gray = apply_semi_transparent_background(
                image=back_image_gray, 
                opacity=0.1,  # 可以调整半透明程度
                bg_color=self.bg_color  # 默认白色背景
            )
            
            back_image_gray.paste(image, (0, 0), image.split()[3] if image.mode == 'RGBA' else None)
            # encode into base64
            final_image = image_to_base64_pil(back_image_gray)
            
            results.append({"id": id, "image": final_image})
            
            gaussians[index].opacity[2] = 0
            
        return results
    
    def render_single_gaussian(self, gaussian_index, device="cuda", bg_color=None, set_to_red=False):
        gaussians = []
        # 复制高斯列表以避免修改原始数据
        for i in range(len(self.gaussians)):
            gaussians.append(self.gaussians[i].copy())
        
        # 将所有高斯的不透明度设为0
        for i in range(len(gaussians)):
            gaussians[i].opacity[2] = 0
            if set_to_red:
                # turn rgb into red
                gaussians[i].color = np.array([1, 0, 0])
            
        # 只设置目标高斯的不透明度为原始值
        gaussians[gaussian_index].opacity[2] = self.gaussians[gaussian_index].opacity[2]
        
        # 收集所有高斯的不透明度
        opacity = None
        for i in range(len(gaussians)):
            if i == 0:
                opacity = np.array([gaussians[i].opacity])
            else:
                opacity = np.vstack([opacity, gaussians[i].opacity])
        
        # 收集所有高斯的颜色
        color = None
        for i in range(len(gaussians)):
            if i == 0:
                color = np.array([gaussians[i].color])
            else:
                color = np.vstack([color, gaussians[i].color])
                
        # 生成传递函数并渲染图像
        tf = self.get_tf(color, opacity)
        image = self.render_image(device, bg_color, force_render=True, 
                                 setImage=False, addRedBorder=False, tf=tf, transparent=True)
        # Get the original background image
        back_image = self.render_image(device)
        # Convert background to grayscale
        back_image_gray = back_image.convert('L').convert('RGB')
        
        back_image_gray = apply_semi_transparent_background(
            image=back_image_gray, 
            opacity=0.1,  # 可以调整半透明程度
            bg_color=self.bg_color  # 默认白色背景
        )
        
        # deprecated Generate the outline image
        # outline_img = self.render_image_with_outline(image)
        outline_img = image
        # Overlay outline_img onto the grayscale background
        # convert outline_img from RGBA to RGB
        back_image_gray.paste(outline_img, (0, 0), outline_img.split()[3] if outline_img.mode == 'RGBA' else None)
        outline_img = back_image_gray
        return outline_img
    
    def outline_active_gaussians(self, device="cuda", bg_color=None, set_to_red=False):
        # 复制高斯列表以避免修改原始数据
        gaussians = [self.gaussians[i].copy() for i in range(len(self.gaussians))]
        
        # 将所有高斯的不透明度设为0
        for i in range(len(gaussians)):
            if set_to_red:
                # turn rgb into red
                gaussians[i].color = np.array([1, 0, 0])
            gaussians[i].opacity[2] = 0
            
        for i in range(self.gmm_num):
            if gaussians[i].is_frozen():
                continue
            # 只设置目标高斯的不透明度为原始值
            gaussians[i].opacity[2] = self.gaussians[i].opacity[2]
        
        # 收集所有高斯的不透明度
        opacity = None
        for i in range(len(gaussians)):
            if i == 0:
                opacity = np.array([gaussians[i].opacity])
            else:
                opacity = np.vstack([opacity, gaussians[i].opacity])
        
        # 收集所有高斯的颜色
        color = None
        for i in range(len(gaussians)):
            if i == 0:
                color = np.array([gaussians[i].color])
            else:
                color = np.vstack([color, gaussians[i].color])
                
        # 生成传递函数并渲染图像
        tf = self.get_tf(color, opacity)
        image = self.render_image(device, bg_color, force_render=True, 
                                setImage=False, addRedBorder=False, tf=tf, transparent=True)
        # Get the original background image
        back_image = self.render_image(device)
        # Convert background to grayscale
        back_image_gray = back_image.convert('L').convert('RGB')
        # 然后降低亮度，使背景更暗
        
        back_image_gray = apply_semi_transparent_background(
            image=back_image_gray, 
            opacity=0.1,  # 可以调整半透明程度
            bg_color=self.bg_color  # 默认白色背景
        )
        # Generate the outline image
        # outline_img = self.render_image_with_outline(image)
        outline_img = image
        # Overlay outline_img onto the grayscale background
        # convert outline_img from RGBA to RGB
        back_image_gray.paste(outline_img, (0, 0), outline_img.split()[3] if outline_img.mode == 'RGBA' else None)
        outline_img = back_image_gray
        return outline_img
    
    
    def plot_tf_tensor(self):
        """
        将传递函数张量可视化并返回为PIL Image
        
        返回:
            PIL.Image: 包含传递函数可视化的PIL图像对象
        """
        
        # 将tf转换为numpy数组以便处理
        tf = self.get_tf()
        tf_np = tf.cpu().numpy()[0]  # 去掉batch维度
        
        # 提取颜色和不透明度
        ctf = tf_np[:, 0:3]  # RGB颜色部分
        otf = tf_np[:, 3]    # 不透明度部分
        
        # 创建x轴数据 (标量值)
        scalar_values = np.linspace(0, 255, len(tf_np))
        # 创建绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TFparamsBase.W/100, TFparamsBase.H/100))

        # 绘制颜色映射
        cmap = np.zeros((1, len(scalar_values), 3))
        for i in range(len(scalar_values)):
            cmap[0, i] = ctf[i]

        ax1.imshow(cmap, aspect='auto', extent=[0, 255, 0, 1])
        ax1.set_title('传递函数：颜色映射')
        ax1.set_xlabel('标量值')
        ax1.set_yticks([])  # 隐藏Y轴刻度

        # 绘制不透明度曲线
        ax2.plot(scalar_values, otf, 'b-', linewidth=2)
        ax2.set_xlim(0, 255)
        ax2.set_ylim(0, max(1.1, otf.max()))
        ax2.set_title('传递函数：不透明度映射')
        ax2.set_xlabel('标量值')
        ax2.set_ylabel('不透明度')

        # 调整布局
        plt.tight_layout()

        # 将图形转换为PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # 关闭图形以释放内存
        
        return img

    def create_rotation_video(self, fps=30, duration=5, resolution=(512, 512)):
        """
        创建旋转视频，可以输出到文件或内存缓冲区
        
        Args:
            output_path: 输出视频文件路径，如果提供output_buffer则忽略
            output_buffer: 输出内存缓冲区，优先于output_path
            fps: 视频帧率
            duration: 视频时长（秒）
            resolution: 视频分辨率
        """
        output_buffer = io.BytesIO()
        # 计算总帧数
        total_frames = fps * duration
        
        # 保存当前相机参数
        original_camera = self.camera
        original_pitch, original_yaw, distance, center, fov = original_camera
        # 使用内存缓冲区
        # 设置视频编码器并创建视频writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或其他编码器如'XVID'
        
        # 使用临时文件，然后读取其内容到缓冲区
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            writer = cv2.VideoWriter(
                temp_file.name, 
                fourcc, 
                fps, 
                resolution
            )
            
            # 生成旋转视频帧
            for frame_idx in range(total_frames):
                # 计算当前角度 (围绕对象旋转一圈)
                angle = frame_idx / total_frames * 2 * np.pi
                
                # 更新相机角度
                self.camera = (original_pitch, original_yaw + angle, distance, center, fov)
                
                # 渲染当前帧
                img = self.render_image(setImage=False, addRedBorder=False,force_render=True)
                # img.save(f'./rotate_img/img{frame_idx}.png')
                # 将PIL Image转换为OpenCV格式
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PIL是RGB，OpenCV是BGR
                
                # 写入帧
                writer.write(img)
                
            # 释放资源
            writer.release()
            
            # 重置文件指针并读取内容到缓冲区
            temp_file.seek(0)
            output_buffer.write(temp_file.read())
            output_buffer.seek(0)
    
        # 恢复原始相机参数
        self.camera = original_camera
        return output_buffer

    def __gen_ctf_from_gmm(self, color_mat, min_scalar_value, max_scalar_value):
        """从高斯混合模型生成颜色传递函数"""
        sort_color_mat = np.unique(color_mat, axis=0)  # 去重并排序
        color_mat[0, 1:] = color_mat[1, 1:]  # 保证第一个颜色点和第二个颜色点相同
        color_mat[-1, 1:] = color_mat[-2, 1:]  # 保证最后一个颜色点和倒数第二个颜色点相同
        cur_color_ind = 0
        color_map = np.zeros((self.tf_size, 4))
        
        for idx in range(self.tf_size):
            interp = idx / (self.tf_size - 1)
            scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
            color_map[idx, 0] = scalar_val
            
            while cur_color_ind < len(sort_color_mat) - 2 and scalar_val > sort_color_mat[cur_color_ind+1, 0]:
                cur_color_ind += 1

            cur_color_sv = sort_color_mat[cur_color_ind, 0]
            next_color_sv = sort_color_mat[cur_color_ind+1, 0]
            
            if next_color_sv == cur_color_sv:
                scalar_val_interp = 0
            else:
                scalar_val_interp = (scalar_val - cur_color_sv) / (next_color_sv - cur_color_sv)
            
            color_map[idx, 1:] = sort_color_mat[cur_color_ind, 1:] + scalar_val_interp * \
                (sort_color_mat[cur_color_ind + 1, 1:] - sort_color_mat[cur_color_ind, 1:])
        
        return color_map
    
    def __gen_otf_from_gmm(self, opacity_gmm, min_scalar_value, max_scalar_value):
        """从高斯混合模型生成不透明度传递函数"""
        opacity_map = np.zeros((self.tf_size, 2))
        for idx in range(self.tf_size):
            interp = float(idx)/(self.tf_size-1)
            scalar_val = min_scalar_value+interp * \
                (max_scalar_value-min_scalar_value)
            gmm_sample = np.sum(opacity_gmm[:, 2] * np.exp(-np.power(
                (scalar_val - opacity_gmm[:, 0]), 2) / np.power(opacity_gmm[:, 1], 2)))
            gmm_sample = max(TFparamsBase.min_opacity, min(TFparamsBase.max_opacity, gmm_sample))
            opacity_map[idx, 0] = scalar_val
            opacity_map[idx, 1] = gmm_sample
        return opacity_map
    
    def __update_x(self, opacity, color):
        """更新颜色点的x坐标以匹配不透明度点"""
        # 这里需要根据具体需求实现
        return opacity, color