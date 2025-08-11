import random


class GeneticConfig:
    def __init__(self, diversity_level='high'):
        """
        Initialize genetic algorithm configuration, setting ratios based on diversity requirements.
        
        :param diversity_level: Diversity requirement ('high', 'optimize', 'fast_converge')
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
        Return corresponding ratio configuration based on different diversity requirements.
        """
        if self.diversity_level == 'high':
            return {
                'elite_retention': 0.05,  # Elite retention ratio: 5%
                'elite_crossover_mutation': (0.15, 0.20),  # Crossover mutation between elites: 15%-20%
                'elite_remaining_crossover_mutation': (0.35, 0.40),  # Crossover mutation between elites and remaining individuals: 35%-40%
                'remaining_crossover_mutation': (0.35, 0.40),  # Random crossover mutation of remaining individuals: 45%-50%
                'random_generation': (0.25, 0.30)
            }
        elif self.diversity_level == 'optimize':
            return {
                'elite_retention': 0.10,  # Elite retention ratio: 10%
                'elite_crossover_mutation': (0.10, 0.15),  # Crossover mutation between elites: 10%-15%
                'elite_remaining_crossover_mutation': (0.25, 0.30),  # Crossover mutation between elites and remaining individuals: 25%-30%
                'remaining_crossover_mutation': (0.40, 0.45),  # Random crossover mutation of remaining individuals: 40%-45%
                'random_generation': (0.15, 0.20)
            }
        elif self.diversity_level == 'fast_converge':
            return {
                'elite_retention': 0.15,  # Elite retention ratio: 15%
                'elite_crossover_mutation': (0.05, 0.10),  # Crossover mutation between elites: 5%-10%
                'elite_remaining_crossover_mutation': (0.20, 0.25),  # Crossover mutation between elites and remaining individuals: 20%-25%
                'remaining_crossover_mutation': (0.40, 0.50),  # Random crossover mutation of remaining individuals: 40%-50%
                'random_generation': (0.05, 0.10)
            }
        else:
            raise ValueError("Unknown diversity_level. Please choose from 'high', 'optimize', or 'fast_converge'.")

    def get_config(self, iter, maxiter):
        """
        Get current configuration dictionary.
        """
        if iter < maxiter * 0.4:
            # First 40% of iterations, use high diversity strategy
            if self.diversity_level != 'high':
                self.update_config('high')
        elif iter < maxiter * 0.7:
            # Middle 30% of iterations, use elite gene propagation strategy
            if self.diversity_level != 'optimize':
                self.update_config('optimize')
        else:
            # Last 30% of iterations, use fast convergence strategy
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
        Update configuration, changing diversity requirement level.
        
        :param diversity_level: New diversity requirement ('high', 'optimize', 'fast_converge')
        """
        self.diversity_level = diversity_level
        self.config = self._get_config()

    def display_config(self):
        """
        Display current configuration ratios.
        """
        print(f"Current diversity level: {self.diversity_level}")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    @staticmethod
    def adaptive_mutate(iter, maxiter, min_scale=0.05, max_scale=0.3):
        """ Adaptive mutation scale, determined by the relationship between min_scale and max_scale whether to increase or decrease with iteration count """
        if min_scale > max_scale:
            # If min_scale is greater than max_scale, mutation scale increases with iteration count
            progress = iter / maxiter
            return max_scale + (min_scale - max_scale) * progress
        else:
            # Original behavior, mutation scale decreases with iteration count
            return max(max_scale * (1 - iter / maxiter), min_scale)

    @staticmethod
    def get_mutation_factor(mutation_factor, iter, maxiter):
        scale_factor = mutation_factor if isinstance(mutation_factor, (int, float)) \
            else GeneticConfig.adaptive_mutate(iter=iter, maxiter=maxiter, min_scale=mutation_factor[0], max_scale=mutation_factor[1])
        
        return scale_factor
    
    def set_mode_specific_factors(self, mode='balanced', intensity=1.5):
        """
        Adjust mutation rates and scales based on specified mode
        
        :param mode: Mutation mode ('balanced', 'shape', 'color', 'position')
        :param intensity: Enhancement intensity multiplier, used to control the enhancement magnitude of mode-specific parameters
        """
        # Save backup of original values (if not already backed up)
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
        
        # First restore to original values
        for key, value in self._original_factors.items():
            setattr(self, key, value)
        
        # Check each mutation parameter type and perform appropriate multiplication
        def safe_multiply(value, multiplier):
            if isinstance(value, (int, float)):
                return value * multiplier
            elif isinstance(value, (tuple, list)):
                # If it's a tuple or list, multiply each element
                return tuple(v * multiplier for v in value)
            else:
                # Other types remain unchanged
                return value
        
        # Adjust mutation parameters based on mode
        if mode == 'shape':
            # Enhance shape-related mutation parameters
            self.op_mutation_rate = safe_multiply(self.op_mutation_rate, intensity)
            self.op_mutation_scale = safe_multiply(self.op_mutation_scale, intensity)
            self.x_mutation_scale = safe_multiply(self.x_mutation_scale, intensity)
            self.bandwidth_mutation_scale = safe_multiply(self.bandwidth_mutation_scale, intensity)
            print(f"Shape mode activated - Enhancing shape-related mutation factors by {intensity}x")
            
        elif mode == 'color':
            # Enhance color-related mutation parameters
            self.color_mutation_rate = safe_multiply(self.color_mutation_rate, intensity)
            self.H_mutation_scale = safe_multiply(self.H_mutation_scale, intensity)
            self.SL_mutation_scale = safe_multiply(self.SL_mutation_scale, intensity)
            print(f"Color mode activated - Enhancing color-related mutation factors by {intensity}x")
            
        elif mode == 'position':
            # Enhance position-related mutation parameters
            self.cam_mutation_rate = safe_multiply(self.cam_mutation_rate, intensity)
            self.cam_mutation_scale = safe_multiply(self.cam_mutation_scale, intensity)
            print(f"Position mode activated - Enhancing position-related mutation factors by {intensity}x")
            
        elif mode == 'balanced':
            # Balanced mode, use original parameters
            print("Balanced mode activated - Using original mutation factors")
            
        else:
            # Enhance all parameters
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
    
        # Optional: print current mutation factor settings
        # self._print_current_factors()

    def _print_current_factors(self):
        """Print current mutation factor settings"""
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
        