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
        self.matches = set()  # Record IDs of opponents battled against
        TFparamsBase.max_opacity = max(TFparamsBase.max_opacity, bound.opacity_bound.y[1])
        TFparamsBase.min_opacity = min(TFparamsBase.min_opacity, bound.opacity_bound.y[0])
            
    def __initialize_from_tfparams(self, tfparams):
        """Initialize from existing tfparams, creating independent data copies"""
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
        Convert TFparams object to JSON serializable dictionary.
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
        self.matches = tfparamsbase.matches  # Record IDs of opponents battled against
        
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
        Update camera rotation angle based on mouse/touch drag
        
        Args:
            direction (dict): Dictionary containing deltaX and deltaY representing drag offset
        """
        # Extract drag offset
        delta_x = direction.get('deltaX', 0)
        delta_y = direction.get('deltaY', 0)
        
        # Define rotation sensitivity coefficient (adjustable as needed)
        sensitivity = 0.5
        
        # Current camera parameters
        pitch, yaw, distance, center, fov = self.camera
        
        # Update yaw angle based on horizontal drag - horizontal drag affects left-right rotation
        # Negative sign because dragging right (positive deltaX) should decrease yaw (camera turns left)
        yaw = (yaw - delta_x * sensitivity) % 360
        
        # Update pitch angle based on vertical drag - vertical drag affects up-down pitch
        # Negative sign because dragging down (positive deltaY) should decrease pitch (camera looks up)
        pitch += delta_y * sensitivity
        
        # Limit pitch range to prevent excessive rotation (usually between -90 and 90 degrees)
        pitch = max(-85, min(85, pitch))
        
        # Update camera parameters
        self.camera = (pitch, yaw, distance, center, fov)
        self.image = None
        self.img_str = None

    def update_camera_zoom(self, zoom_factor=0.1):
        """
        Update camera zoom level based on zoom factor
        
        Args:
            zoom_factor (float): Zoom factor, positive value for zoom in, negative value for zoom out
        """
        # Current camera parameters
        pitch, yaw, distance, center, fov = self.camera
        # Update distance
        distance *= (1 - zoom_factor)
        # Update camera parameters
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
        # Random color for each Gaussian peak coordinate
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
            # Extract camera parameters
            pitch1, yaw1, distance1, center1, fov1 = TF1.camera
            pitch2, yaw2, distance2, center2, fov2 = TF2.camera

            # Create two offspring for crossover
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

            # Perform crossover on parameters of each point
            for i in range(TF1.opacity.shape[0]):  # Iterate through each sample point
                child1_point = []
                child2_point = []
                # Decide whether to swap entire row based on rate
                if random.random() < rate:
                    # Swap entire row
                    child1_point = list(TF2.opacity[i])
                    child2_point = list(TF1.opacity[i])
                else:
                    # Keep original
                    child1_point = list(TF1.opacity[i])
                    child2_point = list(TF2.opacity[i])

                # Store crossover results in offspring
                child1_opacity.append(child1_point)
                child2_opacity.append(child2_point)

            # Convert crossover list to numpy array
            child1.opacity = np.array(child1_opacity)
            child2.opacity = np.array(child2_opacity)

            # Update x values for color points
            child1.__update_x()
            child2.__update_x()
        
        def __cross_color(rate=0.5):
            child1_color = []
            child2_color = []

            # Perform crossover on parameters of each point
            for i in range(TF1.color.shape[0]):  # Iterate through each sample point
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

                # Store crossover results in offspring
                child1_color.append(child1_point)
                child2_color.append(child2_point)

            # Convert crossover list to numpy array
            child1.color = np.array(child1_color)
            child2.color = np.array(child2_color)
            
        def __cross_all_old(rate=0.5):
            # Perform crossover on parameters of each point
            i1 = 0
            i2 = 0
            while((i1 < TF1.gmm_num and i2 < TF2.gmm_num)):  # Iterate through each sample point
                while (i1 < TF1.gmm_num and TF1.gaussians[i1].is_frozen()):
                    i1 += 1
                while (i2 < TF2.gmm_num and TF2.gaussians[i2].is_frozen()):
                    i2 += 1
                
                # Check if indices are still within bound after skipping frozen gaussians
                if i1 >= TF1.gmm_num or i2 >= TF2.gmm_num:
                    break
                    
                # Decide whether to swap entire row based on rate
                if random.random() < rate:
                    # Swap entire row
                    child1.gaussians[i1] = TF2.gaussians[i2].copy()
                    child2.gaussians[i2] = TF1.gaussians[i1].copy()
                i1 += 1
                i2 += 1
                
        def __cross_all(rate=0.6):
            """
            Enhanced Gaussian parameter crossover operation with fine-grained attribute-level crossover
            
            Args:
                rate (float): Base probability for performing crossover
            """
            # Perform crossover on parameters of each point
            i1 = 0
            i2 = 0
            while((i1 < TF1.gmm_num and i2 < TF2.gmm_num)):  # Iterate through each sample point
                while (i1 < TF1.gmm_num and TF1.gaussians[i1].is_frozen()):
                    i1 += 1
                while (i2 < TF2.gmm_num and TF2.gaussians[i2].is_frozen()):
                    i2 += 1
                
                # Check if indices are still within valid range
                if i1 >= TF1.gmm_num or i2 >= TF2.gmm_num:
                    break
                    
                # Decide whether to swap each parameter individually
                cross_decision = {
                    'x': random.random() < rate,
                    'bandwidth': random.random() < rate,
                    'y': random.random() < rate,
                    'color': random.random() < rate
                }
                
                # Execute swap operations based on crossover decisions
                if cross_decision['x']:
                    # Swap x coordinate
                    temp_x = child1.gaussians[i1].opacity[0]
                    child1.gaussians[i1].opacity[0] = child2.gaussians[i2].opacity[0]
                    child2.gaussians[i2].opacity[0] = temp_x
                    
                if cross_decision['bandwidth']:
                    # Swap bandwidth
                    temp_bandwidth = child1.gaussians[i1].opacity[1]
                    child1.gaussians[i1].opacity[1] = child2.gaussians[i2].opacity[1]
                    child2.gaussians[i2].opacity[1] = temp_bandwidth
                    
                if cross_decision['y']:
                    # Swap opacity height
                    temp_y = child1.gaussians[i1].opacity[2]
                    child1.gaussians[i1].opacity[2] = child2.gaussians[i2].opacity[2]
                    child2.gaussians[i2].opacity[2] = temp_y
                    
                if cross_decision['color']:
                    # Swap entire color
                    temp_color = np.copy(child1.gaussians[i1].color)
                    child1.gaussians[i1].color = np.copy(child2.gaussians[i2].color)
                    child2.gaussians[i2].color = temp_color
                    
                # If no parameters are swapped, force swap a random parameter with certain probability
                if not any(cross_decision.values()) and random.random() < 0.5:
                    # Randomly select a parameter to swap
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
            # Calculate mutation scale
            # Determine if mutation_scale is a single value or range tuple
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
                # Randomly swap two colors instead of shuffling all colors
                unfrozen_indices = [i for i in range(len(self.gaussians)) if not self.gaussians[i].is_frozen()]
                
                # Only perform swap when there are at least two unfrozen Gaussians
                if len(unfrozen_indices) >= 2:
                    # Randomly select two different indices
                    idx1, idx2 = random.sample(unfrozen_indices, 2)
                    
                    # Swap colors of these two Gaussians
                    temp_color = np.copy(self.gaussians[idx1].color)
                    self.gaussians[idx1].color = np.copy(self.gaussians[idx2].color)
                    self.gaussians[idx2].color = temp_color
            else:
            # Skip first and last color points
                for i in range(len(self.gaussians)):
                        if self.gaussians[i].is_frozen():
                            continue
                    # Only mutate when random probability is less than mutation rate
                        # Convert from RGB to HLS color space
                        hues, lightnesses, saturations = colorsys.rgb_to_hls(self.gaussians[i].color[0], self.gaussians[i].color[1], self.gaussians[i].color[2])
                        
                        # Randomly select a channel for mutation (0=hue, 1=saturation, 2=lightness)
                        # channel = random.randint(0, 2)
                        # if channel == 0:  # Mutate hue
                        if random.random() < rate_factor:
                            if directions is not None and directions.color.hues is not None:
                                hues = directions.color.hues
                            else:
                                # Perceptually sensitive hue mutation improvement (maintain 0-1 range)
                                hue_span = color_bound.hues[1] - color_bound.hues[0]
                                red_zone = 0.05  # Red sensitive zone boundary (corresponding to 0-0.05 and 0.95-1.0)
                                # Dynamic step size adjustment (red zone step size reduced by 60%)
                                H_scale = np.where(((hues < red_zone) | (hues > 1-red_zone)), 
                                                H_scale_factor * 0.4,  # Red zone
                                                H_scale_factor) * hue_span  # Normal zone
                                # Circular space smooth mutation (avoid sudden changes through polar coordinates)
                                theta = hues * 2 * np.pi  # Map to polar angle
                                dh = np.random.normal(0, H_scale)  # Generate mutation amount
                                hues = ((theta + dh) % (2*np.pi)) / (2*np.pi)  # Circular mapping
                                # Constrain to target range
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
                        
                        # Convert back to RGB
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
        """Generate transfer function tensor"""
        # Sort Gaussian points by opacity x-coordinate
        self.gaussians = sorted(self.gaussians, key=lambda x: x.opacity[0])
        
        if opacity is None:
            for i in range(len(self.gaussians)):
                # Stack opacity
                if i == 0:
                    opacity = np.array([self.gaussians[i].opacity])
                else:
                    opacity = np.vstack([opacity, self.gaussians[i].opacity])
                    
        if color is None:
            for i in range(len(self.gaussians)):
                # Stack color
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
                opacity=0.1,  # Can adjust semi-transparency level
                bg_color=self.bg_color  # Default white background
            )
            
            back_image_gray.paste(image, (0, 0), image.split()[3] if image.mode == 'RGBA' else None)
            # encode into base64
            final_image = image_to_base64_pil(back_image_gray)
            
            results.append({"id": id, "image": final_image})
            
            gaussians[index].opacity[2] = 0
            
        return results
    
    def render_single_gaussian(self, gaussian_index, device="cuda", bg_color=None, set_to_red=False):
        gaussians = []
        # Copy Gaussian list to avoid modifying original data
        for i in range(len(self.gaussians)):
            gaussians.append(self.gaussians[i].copy())
        
        # Set all Gaussian opacities to 0
        for i in range(len(gaussians)):
            gaussians[i].opacity[2] = 0
            if set_to_red:
                # turn rgb into red
                gaussians[i].color = np.array([1, 0, 0])
            
        # Only set target Gaussian opacity to original value
        gaussians[gaussian_index].opacity[2] = self.gaussians[gaussian_index].opacity[2]
        
        # Collect all Gaussian opacities
        opacity = None
        for i in range(len(gaussians)):
            if i == 0:
                opacity = np.array([gaussians[i].opacity])
            else:
                opacity = np.vstack([opacity, gaussians[i].opacity])
        
        # Collect all Gaussian colors
        color = None
        for i in range(len(gaussians)):
            if i == 0:
                color = np.array([gaussians[i].color])
            else:
                color = np.vstack([color, gaussians[i].color])
                
        # Generate transfer function and render image
        tf = self.get_tf(color, opacity)
        image = self.render_image(device, bg_color, force_render=True, 
                                 setImage=False, addRedBorder=False, tf=tf, transparent=True)
        # Get the original background image
        back_image = self.render_image(device)
        # Convert background to grayscale
        back_image_gray = back_image.convert('L').convert('RGB')
        
        back_image_gray = apply_semi_transparent_background(
            image=back_image_gray, 
            opacity=0.1,  # Can adjust semi-transparency level
            bg_color=self.bg_color  # Default white background
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
        # Copy Gaussian list to avoid modifying original data
        gaussians = [self.gaussians[i].copy() for i in range(len(self.gaussians))]
        
        # Set all Gaussian opacities to 0
        for i in range(len(gaussians)):
            if set_to_red:
                # turn rgb into red
                gaussians[i].color = np.array([1, 0, 0])
            gaussians[i].opacity[2] = 0
            
        for i in range(self.gmm_num):
            if gaussians[i].is_frozen():
                continue
            # Only set target Gaussian opacity to original value
            gaussians[i].opacity[2] = self.gaussians[i].opacity[2]
        
        # Collect all Gaussian opacities
        opacity = None
        for i in range(len(gaussians)):
            if i == 0:
                opacity = np.array([gaussians[i].opacity])
            else:
                opacity = np.vstack([opacity, gaussians[i].opacity])
        
        # Collect all Gaussian colors
        color = None
        for i in range(len(gaussians)):
            if i == 0:
                color = np.array([gaussians[i].color])
            else:
                color = np.vstack([color, gaussians[i].color])
                
        # Generate transfer function and render image
        tf = self.get_tf(color, opacity)
        image = self.render_image(device, bg_color, force_render=True, 
                                setImage=False, addRedBorder=False, tf=tf, transparent=True)
        # Get the original background image
        back_image = self.render_image(device)
        # Convert background to grayscale
        back_image_gray = back_image.convert('L').convert('RGB')
        # Then reduce brightness to make background darker
        
        back_image_gray = apply_semi_transparent_background(
            image=back_image_gray, 
            opacity=0.1,  # Can adjust semi-transparency level
            bg_color=self.bg_color  # Default white background
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
        Visualize transfer function tensor and return as PIL Image
        
        Returns:
            PIL.Image: PIL image object containing transfer function visualization
        """
        
        # Convert tf to numpy array for processing
        tf = self.get_tf()
        tf_np = tf.cpu().numpy()[0]  # Remove batch dimension
        
        # Extract color and opacity
        ctf = tf_np[:, 0:3]  # RGB color part
        otf = tf_np[:, 3]    # Opacity part
        
        # Create x-axis data (scalar values)
        scalar_values = np.linspace(0, 255, len(tf_np))
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TFparamsBase.W/100, TFparamsBase.H/100))

        # Draw color mapping
        cmap = np.zeros((1, len(scalar_values), 3))
        for i in range(len(scalar_values)):
            cmap[0, i] = ctf[i]

        ax1.imshow(cmap, aspect='auto', extent=[0, 255, 0, 1])
        ax1.set_title('Transfer Function: Color Mapping')
        ax1.set_xlabel('Scalar Value')
        ax1.set_yticks([])  # Hide Y-axis ticks

        # Draw opacity curve
        ax2.plot(scalar_values, otf, 'b-', linewidth=2)
        ax2.set_xlim(0, 255)
        ax2.set_ylim(0, max(1.1, otf.max()))
        ax2.set_title('Transfer Function: Opacity Mapping')
        ax2.set_xlabel('Scalar Value')
        ax2.set_ylabel('Opacity')

        # Adjust layout
        plt.tight_layout()

        # Convert figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)  # Close figure to release memory
        
        return img

    def create_rotation_video(self, fps=30, duration=5, resolution=(512, 512)):
        """
        Create rotation video, can output to file or memory buffer
        
        Args:
            output_path: Output video file path, ignored if output_buffer is provided
            output_buffer: Output memory buffer, takes priority over output_path
            fps: Video frame rate
            duration: Video duration (seconds)
            resolution: Video resolution
        """
        output_buffer = io.BytesIO()
        # Calculate total frames
        total_frames = fps * duration
        
        # Save current camera parameters
        original_camera = self.camera
        original_pitch, original_yaw, distance, center, fov = original_camera
        # Use memory buffer
        # Set up video encoder and create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or other encoders like 'XVID'
        
        # Use temporary file, then read its content to buffer
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            writer = cv2.VideoWriter(
                temp_file.name, 
                fourcc, 
                fps, 
                resolution
            )
            
            # Generate rotation video frames
            for frame_idx in range(total_frames):
                # Calculate current angle (rotate around object once)
                angle = frame_idx / total_frames * 2 * np.pi
                
                # Update camera angle
                self.camera = (original_pitch, original_yaw + angle, distance, center, fov)
                
                # Render current frame
                img = self.render_image(setImage=False, addRedBorder=False,force_render=True)
                # img.save(f'./rotate_img/img{frame_idx}.png')
                # Convert PIL Image to OpenCV format
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PIL is RGB, OpenCV is BGR
                
                # Write frame
                writer.write(img)
                
            # Release resources
            writer.release()
            
            # Reset file pointer and read content to buffer
            temp_file.seek(0)
            output_buffer.write(temp_file.read())
            output_buffer.seek(0)
    
        # Restore original camera parameters
        self.camera = original_camera
        return output_buffer

    def __gen_ctf_from_gmm(self, color_mat, min_scalar_value, max_scalar_value):
        """Generate color transfer function from Gaussian mixture model"""
        sort_color_mat = np.unique(color_mat, axis=0)  # Remove duplicates and sort
        color_mat[0, 1:] = color_mat[1, 1:]  # Ensure first and second color points are the same
        color_mat[-1, 1:] = color_mat[-2, 1:]  # Ensure last and second-to-last color points are the same
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
        """Generate opacity transfer function from Gaussian mixture model"""
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
        """Update x-coordinates of color points to match opacity points"""
        # Implementation needed based on specific requirements
        return opacity, color