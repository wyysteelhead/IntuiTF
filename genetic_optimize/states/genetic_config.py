import random


class GeneticConfig:
    def __init__(self, diversity_level='high'):
        """
        初始化遗传算法配置，根据多样性需求设置比例。
        
        :param diversity_level: 多样性需求 ('high', 'optimize', 'fast_converge')
        """
        self.diversity_level = diversity_level
        self.config = self._get_config()
        self.cam_mutation_rate = 1.0
        self.op_mutation_rate = 1.0
        self.color_mutation_rate = 1.0
        self.mutation_scale = 1.0
        self.H_mutation_scale = 1.0
        self.text_mode = False
    
    def setFactor(self, cam_mutation_rate, cam_mutation_scale, op_mutation_rate, op_mutation_scale, x_mutation_scale, bandwidth_mutation_scale, color_mutation_rate, H_mutation_scale, SL_mutation_scale):
        self.cam_mutation_rate = cam_mutation_rate
        self.cam_mutation_scale = cam_mutation_scale
        self.op_mutation_rate = op_mutation_rate
        self.op_mutation_scale = op_mutation_scale
        self.x_mutation_scale = x_mutation_scale
        self.bandwidth_mutation_scale = bandwidth_mutation_scale
        self.color_mutation_rate = color_mutation_rate
        self.H_mutation_scale = H_mutation_scale
        self.SL_mutation_scale = SL_mutation_scale

    def _get_config(self):
        """
        根据不同的多样性需求返回相应的比例配置。
        """
        if self.diversity_level == 'high':
            return {
                'elite_retention': 0.05,  # 精英保留比例：5%
                'elite_crossover_mutation': (0.15, 0.20),  # 精英之间的交叉变异：15%-20%
                'elite_remaining_crossover_mutation': (0.35, 0.40),  # 精英与剩余个体交叉变异：35%-40%
                'remaining_crossover_mutation': (0.35, 0.40),  # 剩余个体的随机交叉变异：45%-50%
                'random_generation': (0.25, 0.30)
            }
        elif self.diversity_level == 'optimize':
            return {
                'elite_retention': 0.10,  # 精英保留比例：10%
                'elite_crossover_mutation': (0.10, 0.15),  # 精英之间的交叉变异：10%-15%
                'elite_remaining_crossover_mutation': (0.25, 0.30),  # 精英与剩余个体交叉变异：25%-30%
                'remaining_crossover_mutation': (0.40, 0.45),  # 剩余个体的随机交叉变异：40%-45%
                'random_generation': (0.15, 0.20)
            }
        elif self.diversity_level == 'fast_converge':
            return {
                'elite_retention': 0.15,  # 精英保留比例：15%
                'elite_crossover_mutation': (0.05, 0.10),  # 精英之间的交叉变异：5%-10%
                'elite_remaining_crossover_mutation': (0.20, 0.25),  # 精英与剩余个体交叉变异：20%-25%
                'remaining_crossover_mutation': (0.40, 0.50),  # 剩余个体的随机交叉变异：40%-50%
                'random_generation': (0.05, 0.10)
            }
        else:
            raise ValueError("Unknown diversity_level. Please choose from 'high', 'optimize', or 'fast_converge'.")

    def get_config(self, iter, maxiter):
        """
        获取当前的配置字典。
        """
        if iter < maxiter * 0.4:
            # 前40%的迭代，使用高多样性策略
            if self.diversity_level != 'high':
                self.update_config('high')
        elif iter < maxiter * 0.7:
            # 中间30%的迭代，使用精英基因传播策略
            if self.diversity_level != 'optimize':
                self.update_config('optimize')
        else:
            # 后30%的迭代，使用快速收敛策略
            if self.diversity_level != 'fast_converge':
                self.update_config('fast_converge')

        elite_retention = self.config['elite_retention']
        # elite_crossover_mutation = random.uniform(self.config['elite_crossover_mutation'][0], self.config['elite_crossover_mutation'][1])
        # elite_remaining_crossover_mutation = random.uniform(self.config['elite_remaining_crossover_mutation'][0], self.config['elite_remaining_crossover_mutation'][1])
        # remaining_crossover_mutation = random.uniform(self.config['remaining_crossover_mutation'][0], self.config['remaining_crossover_mutation'][1])
        random_generation = random.uniform(self.config['random_generation'][0], self.config['random_generation'][1])
        return elite_retention, random_generation

    def update_config(self, diversity_level):
        """
        更新配置，改变多样性需求级别。
        
        :param diversity_level: 新的多样性需求 ('high', 'optimize', 'fast_converge')
        """
        self.diversity_level = diversity_level
        self.config = self._get_config()

    def display_config(self):
        """
        显示当前配置的比例。
        """
        print(f"Current diversity level: {self.diversity_level}")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    @staticmethod
    def adaptive_mutate(iter, maxiter, min_scale=0.05, max_scale=0.3):
        """ 自适应变异尺度，根据min_scale和max_scale的大小关系决定随迭代次数增大或减小 """
        if min_scale > max_scale:
            # 如果min_scale大于max_scale，则变异尺度随迭代次数增大
            progress = iter / maxiter
            return max_scale + (min_scale - max_scale) * progress
        else:
            # 原行为，变异尺度随迭代次数减小
            return max(max_scale * (1 - iter / maxiter), min_scale)

    @staticmethod
    def get_mutation_factor(mutation_factor, iter, maxiter):
        scale_factor = mutation_factor if isinstance(mutation_factor, (int, float)) \
            else GeneticConfig.adaptive_mutate(iter=iter, maxiter=maxiter, min_scale=mutation_factor[0], max_scale=mutation_factor[1])
        
        return scale_factor
    
    def set_mode_specific_factors(self, mode='balanced', intensity=1.5):
        """
        根据指定模式调整变异率和变异尺度
        
        :param mode: 变异模式 ('balanced', 'shape', 'color', 'position')
        :param intensity: 强化程度倍数，用于控制模式特定参数的增强幅度
        """
        # 保存原始值的备份（如果尚未备份）
        if not hasattr(self, '_original_factors'):
            self._original_factors = {
                'cam_mutation_rate': self.cam_mutation_rate,
                'cam_mutation_scale': self.cam_mutation_scale,
                'op_mutation_rate': self.op_mutation_rate,
                'op_mutation_scale': self.op_mutation_scale,
                'x_mutation_scale': self.x_mutation_scale,
                'bandwidth_mutation_scale': self.bandwidth_mutation_scale,
                'color_mutation_rate': self.color_mutation_rate,
                'H_mutation_scale': self.H_mutation_scale,
                'SL_mutation_scale': self.SL_mutation_scale
            }
        
        # 先恢复到原始值
        for key, value in self._original_factors.items():
            setattr(self, key, value)
        
        # 检查各变异参数类型并进行适当的乘法
        def safe_multiply(value, multiplier):
            if isinstance(value, (int, float)):
                return value * multiplier
            elif isinstance(value, (tuple, list)):
                # 如果是元组或列表，对每个元素进行乘法
                return tuple(v * multiplier for v in value)
            else:
                # 其他类型不变
                return value
        
        # 根据模式调整变异参数
        if mode == 'shape':
            # 增强与形状相关的变异参数
            self.op_mutation_rate = safe_multiply(self.op_mutation_rate, intensity)
            self.op_mutation_scale = safe_multiply(self.op_mutation_scale, intensity)
            self.x_mutation_scale = safe_multiply(self.x_mutation_scale, intensity)
            self.bandwidth_mutation_scale = safe_multiply(self.bandwidth_mutation_scale, intensity)
            print(f"Shape mode activated - Enhancing shape-related mutation factors by {intensity}x")
            
        elif mode == 'color':
            # 增强与颜色相关的变异参数
            self.color_mutation_rate = safe_multiply(self.color_mutation_rate, intensity)
            self.H_mutation_scale = safe_multiply(self.H_mutation_scale, intensity)
            self.SL_mutation_scale = safe_multiply(self.SL_mutation_scale, intensity)
            print(f"Color mode activated - Enhancing color-related mutation factors by {intensity}x")
            
        elif mode == 'position':
            # 增强与位置相关的变异参数
            self.cam_mutation_rate = safe_multiply(self.cam_mutation_rate, intensity)
            self.cam_mutation_scale = safe_multiply(self.cam_mutation_scale, intensity)
            print(f"Position mode activated - Enhancing position-related mutation factors by {intensity}x")
            
        elif mode == 'balanced':
            # 平衡模式，使用原始参数
            print("Balanced mode activated - Using original mutation factors")
            
        else:
            # 全部都增强
            self.cam_mutation_rate = safe_multiply(self.cam_mutation_rate, intensity)
            self.cam_mutation_scale = safe_multiply(self.cam_mutation_scale, intensity)
            self.op_mutation_rate = safe_multiply(self.op_mutation_rate, intensity)
            self.op_mutation_scale = safe_multiply(self.op_mutation_scale, intensity)
            self.x_mutation_scale = safe_multiply(self.x_mutation_scale, intensity)
            self.bandwidth_mutation_scale = safe_multiply(self.bandwidth_mutation_scale, intensity)
            self.color_mutation_rate = safe_multiply(self.color_mutation_rate, intensity)
            self.H_mutation_scale = safe_multiply(self.H_mutation_scale, intensity)
            self.SL_mutation_scale = safe_multiply(self.SL_mutation_scale, intensity)
            print(f"All modes activated - Enhancing all mutation factors by {intensity}x")
    
        # 可选：打印当前设置的变异因子
        # self._print_current_factors()

    def _print_current_factors(self):
        """打印当前的变异因子设置"""
        print("Current mutation factors:")
        print(f"- Camera mutation rate: {self.cam_mutation_rate}")
        print(f"- Camera mutation scale: {self.cam_mutation_scale}")
        print(f"- Opacity mutation rate: {self.op_mutation_rate}")
        print(f"- Opacity mutation scale: {self.op_mutation_scale}")
        print(f"- X position mutation scale: {self.x_mutation_scale}")
        print(f"- Bandwidth mutation scale: {self.bandwidth_mutation_scale}")
        print(f"- Color mutation rate: {self.color_mutation_rate}")
        print(f"- Hue mutation scale: {self.H_mutation_scale}")
        print(f"- Saturation/Lightness mutation scale: {self.SL_mutation_scale}")
        