import os
import sys
import time
import pickle
import random
import torch
from tqdm import tqdm
from diffdvr.settings import Settings
from diffdvr import renderer_dtype_torch, renderer_dtype_np
from genetic_optimize.eval.metric_eval import MetricEvaluator
from genetic_optimize.eval.llm_eval import LLM_Evaluator
from genetic_optimize.states.bound import Bound
from genetic_optimize.states.genetic_config import GeneticConfig
from genetic_optimize.config.config_manager import ConfigManager
from genetic_optimize.visualize.gaussian_visualizer import GaussianVisualizer
import numpy as np
import colorsys
import json
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import io
from genetic_optimize.utils.thread import ParallelExecutor
import pyrenderer
import multiprocessing
from genetic_optimize.TFparamsImp import TFparamsImp
import math
from typing import List
import bisect
from genetic_optimize.eval.elo_rating import elo_update, swiss_pairing
from genetic_optimize.utils.image_utils import combine_image, concat_images, compute_ssim, fitness_sharing_with_matrix
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import ast
from genetic_optimize.TFparamsBase import TFparamsBase
			
#是否生成高斯分布GIF
Test=False

class GeneticAlgorithm:
    """
    A genetic algorithm implementation for optimizing transfer functions in volume rendering.
    
    This class implements a genetic algorithm that evolves a population of transfer functions
    to find optimal visualization parameters for volume rendering. It supports various mutation
    strategies, crossover operations, and evaluation methods including LLM-based evaluation.
    
    Attributes:
        config_manager (ConfigManager): Manages configuration settings for the algorithm
        bound (Bound): Defines parameter boundaries for the genetic operations
        bg_color (tuple): Background color for rendering
        save_path (str): Path to save outputs
        config (dict): Configuration dictionary
        tf_size (int): Size of the transfer function
        population (list): List of individuals in the current population
        iteration (int): Current iteration count
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(
        self, 
        config_manager: ConfigManager,
        population = None, 
        iteration = 0, 
        device="cuda",
        renderer_type="diffdvr"
    ):
        """
        Initialize the genetic algorithm with configuration and population.

        Args:
            config_manager (ConfigManager): Configuration manager instance
            population (list, optional): Initial population. If None, creates new population
            iteration (int, optional): Starting iteration number. Defaults to 0
            device (str, optional): Computing device. Defaults to "cuda"
            renderer_type (str, optional): Type of renderer to use ('diffdvr' or 'anari'). Defaults to "diffdvr"
        """
        # Set renderer-specific dtypes
        self.renderer_dtype_torch, self.renderer_dtype_np = get_renderer_dtypes(renderer_type)
        
        # Dynamically import renderer-specific modules
        if renderer_type == "diffdvr":
            import torch
            from diffdvr.settings import Settings
            import pyrenderer
            from genetic_optimize.TFparamsBase import TFparamsBase
            self.TFparamsClass = TFparamsBase
            self.Settings = Settings
        elif renderer_type == "anari":
            # the anari module is not ready yet, we are still working on it
            from genetic_optimize.TFparamsAnariImp import TFparamsAnariImp
            from anari.settings import AnariSettings
            self.TFparamsClass = TFparamsAnariImp
            self.Settings = AnariSettings  # Will be handled by TFparamsAnariImp
        else:
            raise ValueError(f"Unsupported renderer type: {renderer_type}")

        self.config_manager = config_manager
        self.bound = config_manager.get_bound()
        self.bg_color = config_manager.get_algorithm_config().bg_color
        self.save_path = config_manager.get_algorithm_config().save_path
        self.config = self.bound.config
        self.tf_size = self.bound.opacity_bound.x[1]
        self.renderer_type = renderer_type  # 存储渲染器类型
        
        algorithm_config = config_manager.get_algorithm_config()
        mutation_config = config_manager.get_mutation_config()
        api_config = config_manager.get_api_config()
        
        self.llm_evaluator = LLM_Evaluator(
            base_url=api_config.base_url, 
            api_key=api_config.api_key, 
            prompt_folder=api_config.prompt_folder, 
            quality_metrics=api_config.quality_metrics,
            text_metrics=api_config.text_metrics,
            instruct_number=api_config.instruct_number,
            model_name=algorithm_config.model_name)
            
        if hasattr(algorithm_config, 'style_img_path') and algorithm_config.style_img_path:
            style_image = Image.open(algorithm_config.style_img_path)
            if style_image.mode != 'RGB':
                style_image = style_image.convert('RGB')
            style_image.load()
            style_image_copy = style_image.copy()
            self.llm_evaluator.set_middle_image(style_image_copy)
            
        # self.metric_evaluator = MetricEvaluator()
        self.iteration = iteration
        self.save_interval = algorithm_config.save_interval
        
        self.setting = self.Settings(self.bound.config_file)
        self.volume = self.setting.load_dataset()
        if self.renderer_type == "diffdvr":
            self.gradient = self.setting.load_gradient()
            self.volume.copy_to_gpu()
            if self.gradient:
                self.gradient.copy_to_gpu()
        else:
            self.gradient = None
            
        self.device = device
        self.iter_bias = 0
        self.gaussian_visualizer = GaussianVisualizer()
        self.mode = "quality"
        self.genetic_config = GeneticConfig(diversity_level='high')
        self.genetic_config.setFactor(
            cam_mutation_rate=mutation_config.cam_mutation_rate, 
            cam_mutation_scale=mutation_config.cam_mutation_scale,
            op_mutation_rate=mutation_config.op_mutation_rate, 
            op_mutation_scale=mutation_config.op_mutation_scale,
            x_mutation_scale=mutation_config.x_mutation_scale,
            bandwidth_mutation_scale=mutation_config.bandwidth_mutation_scale,
            color_mutation_rate=mutation_config.color_mutation_rate, 
            H_mutation_scale=mutation_config.H_mutation_scale,
            SL_mutation_scale=mutation_config.SL_mutation_scale
            )
        self.population_size = algorithm_config.population_size
        
        if population is not None:
            self.population = population
            self.population_size = len(population)
            for i in range(len(self.population)):
                population[i].load_render_settings(bound=self.bound, volume=self.volume, gradient=self.gradient, step_size=self.setting.get_stepsize() if self.setting else None, bg_color=self.bg_color)
        else:
            self.population = [self.TFparamsClass(id=id, bound=self.bound, volume=self.volume, gradient=self.gradient, step_size=self.setting.get_stepsize() if self.setting else None, bg_color=self.bg_color, device=self.device, renderer_dtype_np=self.renderer_dtype_np, setInputs=True) for id in range(self.population_size)]
        
        #for backend
        self.TF_static = (self.TFparamsClass.global_inputs, self.TFparamsClass.W, self.TFparamsClass.H, self.TFparamsClass.max_opacity, self.TFparamsClass.min_opacity, self.TFparamsClass.bg_color)
    
    @staticmethod
    def save_state(population, iteration, mode, filename):
        """
        Save the current state of the genetic algorithm.

        Args:
            population (list): Current population
            iteration (int): Current iteration number
            mode (str): Current operation mode
            filename (str): Path to save the state
        """
        # Get the class of the first population member to determine which TFparams class to use
        TFparamsClass = type(population[0])
        pickled_population = [TFparamsClass(id=ind.id, tfparams=ind) for ind in population]
        with open(filename, 'wb') as f:
            pickle.dump({'population': pickled_population, 'iteration': iteration, 'mode': mode}, f)

    @staticmethod
    def load_state(filename='population_state.pkl'):
        """
        Load a previously saved state of the genetic algorithm.

        Args:
            filename (str, optional): Path to the state file. Defaults to 'population_state.pkl'

        Returns:
            tuple: (population, iteration, mode)
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        mode = data.get('mode') if data.get('mode') is not None else "quality"
        return data['population'], data['iteration'], mode
    
    def reset_text(self, text):
        """Reset the text prompt for LLM evaluation."""
        self.llm_evaluator.reset_text(text)
        
    def reset_modification(self, modification):
        """Reset the modification parameters for LLM evaluation."""
        self.llm_evaluator.reset_modification(modification)
    
    def evaluator(self, parent1: TFparamsBase, parent2: TFparamsBase, mode, log_path = ""):
        """
        Compare two individuals using LLM-based evaluation.

        Args:
            parent1 (TFparamsBase): First individual to compare
            parent2 (TFparamsBase): Second individual to compare
            mode (str): Evaluation mode ('finetune', 'quality', or 'text')
            log_path (str, optional): Path to save evaluation logs. Defaults to ""

        Returns:
            tuple: (winner, output, img1_base64, img2_base64)
        """
        if Test:
            return random.choice([parent1,parent2])
        
        if mode == "finetune":
            img1 = parent1.outline_active_gaussians(set_to_red=False)
            img2 = parent2.outline_active_gaussians(set_to_red=False)
        else:
            img1 = parent1.render_image()
            img2 = parent2.render_image()
            
        result, output = self.llm_evaluator.compare_2_image(img1=img1, img2=img2, volume_name=self.config["volume_name"], mode=mode, log_path = log_path)
        if result == "draw":
            result = "draw"
        elif result == "error":
            result = "error"
        if result == "img1":
            winner = parent1
        elif result == "img2":
            winner = parent2
        else:
            winner = None
        output["img1"] = parent1.id
        output["img2"] = parent2.id
        if mode == "finetune" or mode == "image":
            img1_base64, img2_base64 = combine_image(img1, img2, self.llm_evaluator.middle_img)
        else:
            img1_base64, img2_base64 = combine_image(img1, img2)
        return winner, output, img1_base64, img2_base64
    
    def evaluate_gaussian(self, individual: TFparamsBase, i, path):
        """
        Evaluate a single Gaussian component of an individual.

        Args:
            individual (TFparamsBase): Individual to evaluate
            i (int): Index of the Gaussian component
            path (str): Path to save evaluation results

        Returns:
            tuple: (result, output) where result indicates if the Gaussian should be frozen
        """
        img = individual.render_single_gaussian(i, set_to_red=True)
        base_img = individual.render_image()
        result, output = self.llm_evaluator.evaluate_gaussian_image(img=img, base_img=base_img, volume_name=self.config["volume_name"], log_path=path)
        output["tf"] = individual.id
        output["gaussian"] = i
        return result, output

    def fitness_evaluator(self, individual: TFparamsBase):
        """
        Compute fitness score for an individual using metric-based evaluation.

        Args:
            individual (TFparamsBase): Individual to evaluate

        Returns:
            float: Fitness score
        """
        img = individual.image
        fitness = self.metric_evaluator.compute(img)
        return fitness

    def tournament_selection(self, population, mode, evaluator, log_path=""):
        """
        Perform tournament selection between two randomly selected individuals.

        Args:
            population (list): Population to select from
            mode (str): Evaluation mode
            evaluator (callable): Function to evaluate individuals
            log_path (str, optional): Path to save evaluation logs. Defaults to ""

        Returns:
            TFparamsBase: Winner of the tournament
        """
        parent1, parent2 = random.sample(population, 2)
        return evaluator(parent1, parent2, mode, log_path)
    
    @staticmethod
    def gaussian(x, x0, bandwidth, y):
        """
        Compute Gaussian function value.

        Args:
            x (float): Input value
            x0 (float): Mean of the Gaussian
            bandwidth (float): Standard deviation of the Gaussian
            y (float): Amplitude of the Gaussian

        Returns:
            float: Gaussian function value
        """
        return y * np.exp(-((x-x0)**2)/(2*bandwidth**2))
    
    def gaussian_of_population(self, population, min_scalar=0, max_scalar=255, num_points=1000):
        """
        Visualize Gaussian distributions of the entire population.

        Args:
            population (list): Population to visualize
            min_scalar (int, optional): Minimum scalar value. Defaults to 0
            max_scalar (int, optional): Maximum scalar value. Defaults to 255
            num_points (int, optional): Number of points for visualization. Defaults to 1000
        """
        self.gaussian_visualizer.plot_gaussian_of_population(population, min_scalar, max_scalar, num_points)
        
    def plot_gaussian_of_individual(self, individual, min_scalar=0, max_scalar=255, num_points=1000):
        """
        Create a visualization of an individual's Gaussian distributions.
        
        Args:
            individual (TFparamsBase): Individual to visualize
            min_scalar (int, optional): Minimum scalar value. Defaults to 0
            max_scalar (int, optional): Maximum scalar value. Defaults to 255
            num_points (int, optional): Number of points for visualization. Defaults to 1000

        Returns:
            Image: Combined image of Gaussian visualization and caption
        """
        caption = []
        title = f'Gaussian Distribution of Individual (ID: {individual.id})'
        if hasattr(individual, 'parent_id') and individual.parent_id:
            parent_ids_str = ', '.join(map(str, individual.parent_id))
            title += f' | Parents: {parent_ids_str}'
        caption.append(title)
        
        for i, (x0, bandwidth, y) in enumerate(individual.opacity):
            gaussian_info = f"Gaussian {i+1}: x0={x0:.1f}, sigma={bandwidth:.1f}, y={y:.1f}"
            caption.append(gaussian_info)
        
        caption_text = "\n".join(caption)
        font_size = 12
        line_padding = 5
        num_lines = len(caption)
        line_height = font_size + line_padding
        img_height = num_lines * line_height + 20
        img_width = TFparamsBase.W

        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        y_position = 10
        for line in caption:
            try:
                ascii_line = line.encode('ascii', 'replace').decode('ascii')
                draw.text((10, y_position), ascii_line, fill="black", font=font)
            except Exception:
                draw.text((10, y_position), line.encode('ascii', 'replace').decode('ascii'), fill="black", font=font)
            y_position += line_height
        
        tf_img = individual.plot_tf_tensor()
        
        combined_img = Image.new('RGB', (max(img.width, tf_img.width), img.height + tf_img.height))
        combined_img.paste(img, (0, 0))
        combined_img.paste(tf_img, (0, img.height))
        
        return combined_img
        
    def gen_gif_from_images(self, output_gif_path="graph.gif", duration=500, loop=2):
        """
        Generate an animated GIF from the population's Gaussian visualizations.

        Args:
            output_gif_path (str, optional): Path to save the GIF. Defaults to "graph.gif"
            duration (int, optional): Duration for each frame in ms. Defaults to 500
            loop (int, optional): Number of times to loop the GIF. Defaults to 2
        """
        self.gaussian_visualizer.gen_gif_from_images(output_gif_path, duration, loop)
        
    def __crossover_task(self, task):
        """
        Helper function for parallel crossover operations.

        Args:
            task (tuple): Tuple containing (population1, population2)

        Returns:
            TFparamsBase: Child individual from crossover
        """
        population1, population2 = task
        parent1 = random.choice(population1)
        parent2 = random.choice(population2)
        childs = TFparamsBase.crossover(parent1, parent2)
        return childs
        
    def population_crossover(self, population, count=None, num_workers=None):
        """
        Perform crossover operations on the population in parallel.

        Args:
            population (list): Population to perform crossover on
            count (int, optional): Number of crossover operations. Defaults to population size // 2
            num_workers (int, optional): Number of parallel workers. Defaults to None

        Returns:
            list: New population after crossover
        """
        if len(population) <= 1: return population
        new_population = []
        mid = len(population) // 2
        if count == None:
            count = mid
        random.shuffle(population)
        population1, population2 = population[:mid], population[mid:]
        executor = ParallelExecutor(num_workers=num_workers)
        tasks = [(population1, population2) for _ in range(count)]
        new_population = executor.execute(
            tasks=tasks,
            task_fn=self.__crossover_task
        )
        return new_population

    def population_mutation(self, population, bound, iter, max_iter, num_workers = None):
        """
        Perform mutation operations on the population in parallel.

        Args:
            population (list): Population to mutate
            bound (Bound): Parameter boundaries
            iter (int): Current iteration
            max_iter (int): Maximum iterations
            num_workers (int, optional): Number of parallel workers. Defaults to None

        Returns:
            list: Mutated population
        """
        executor = ParallelExecutor(num_workers=num_workers)
        tasks = [(individual, bound, self.genetic_config, iter, max_iter) for individual in population]
        new_population = executor.execute(
            tasks=tasks, 
            task_fn=self.__mutate_task
        )
        return new_population
    
    def __mutate_task(self, task):
        """
        Helper function for parallel mutation operations.

        Args:
            task (tuple): Tuple containing (individual, bound, genetic_config, iter, max_iter)

        Returns:
            TFparamsBase: Mutated individual
        """
        individual, bound, genetic_config, iter, max_iter = task
        individual.mutate(bound=bound, genetic_config=genetic_config, iter=iter, maxiter=max_iter)
        return individual
      
    def save_population_images(self, population, save_path):
        """
        Save rendered images and transfer function visualizations of the population.

        Args:
            population (list): Population to save images for
            save_path (str): Path to save the images
        """
        images = [population[i].render_image() for i in range(len(population))]
        tf_images = [self.plot_gaussian_of_individual(population[i]) for i in range(len(population))]
        if images:
            new_image = concat_images(images)
            new_tf_image = concat_images(tf_images)
            new_full_image = Image.new('RGB', (new_image.width, new_image.height + new_tf_image.height))
            new_full_image.paste(new_image, (0, 0))
            new_full_image.paste(new_tf_image, (0, new_image.height))
            try:
                new_full_image.save(save_path)
            except Exception:
                print("path don't exist, file not saved.")
                
    def save_activate_gasussian_images(self, population, save_path):
        """
        Save visualizations of active Gaussians for the population.

        Args:
            population (list): Population to save images for
            save_path (str): Path to save the images
        """
        images = [population[i].outline_active_gaussians() for i in range(len(population))]
        tf_images = [self.plot_gaussian_of_individual(population[i]) for i in range(len(population))]
        if images:
            new_image = concat_images(images)
            new_tf_image = concat_images(tf_images)
            new_full_image = Image.new('RGB', (new_image.width, new_image.height + new_tf_image.height))
            new_full_image.paste(new_image, (0, 0))
            new_full_image.paste(new_tf_image, (0, new_image.height))
            try:
                new_full_image.save(save_path)
            except Exception:
                print("path don't exist, file not saved.")
                
    def parallel_render_population_image(self, population, num_workers=1):
        """
        Render images for the entire population in parallel.

        Args:
            population (list): Population to render
            num_workers (int, optional): Number of parallel workers. Defaults to None
        """
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(ind.render_image) for ind in population]
                progress_bar = tqdm(total=len(futures), desc="🎨 Rendering Images",
                                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")
                try:
                    for _ in as_completed(futures):
                        progress_bar.update(1) 
                except Exception as e:
                    print(f"[ERROR] Error during rendering: {str(e)}")
                    raise
                finally:
                    progress_bar.close()
                    
        except Exception as e:
            print(f"[ERROR] Parallel rendering failed: {str(e)}")
            raise
        finally:
            if hasattr(self, '_executor'):
                self._executor = None

    def child_generation(self, parent1, parent2, bound, generations):
        """
        Generate a child individual from two parents.

        Args:
            parent1 (TFparamsBase): First parent
            parent2 (TFparamsBase): Second parent
            bound (Bound): Parameter boundaries
            generations (int): Total number of generations

        Returns:
            TFparamsBase: Child individual
        """
        child = GeneticAlgorithm.crossover(parent1, parent2)
        child = GeneticAlgorithm.mutate(individual=child, bound=bound, iter=self.iteration, maxiter=generations)
        return child
            
    def _parallel_child_generation(self, population, mode, bound, generations, log_path=""):
        """
        Generate child individuals in parallel with error handling.

        Args:
            population (list): Population to generate children from
            mode (str): Evaluation mode
            bound (Bound): Parameter boundaries
            generations (int): Total number of generations
            log_path (str, optional): Path to save logs. Defaults to ""

        Returns:
            TFparamsBase: Generated child individual
        """
        parent1 = self.tournament_selection(population, mode, self.evaluator, log_path)
        parent2 = self.tournament_selection(population, mode, self.evaluator, log_path)
        child = self.child_generation(
            parent1, parent2, bound, generations
        )
        return child
    
    def elo_matching(self, p1, p2, mode, save_path):
        """
        Perform Elo-based comparison between two individuals.

        Args:
            p1 (TFparamsBase): First individual
            p2 (TFparamsBase): Second individual
            mode (str): Evaluation mode
            save_path (str): Path to save results

        Returns:
            dict: Match results including winner, loser, output, and image data
        """
        winner, output, img1_base64, img2_base64 = self.evaluator(p1, p2, mode, save_path)
        if winner is None:
            loser = None
        else:
            loser = p1 if winner is p2 else p2
        if output is None: output = ""
        return {"winner": winner, "loser": loser, "output": output, "img1_base64": img1_base64, "img2_base64": img2_base64}
        
    def check_gaussian(self, individual, i, save_path):
        """
        Check if a Gaussian component should be frozen.

        Args:
            individual (TFparamsBase): Individual to check
            i (int): Index of Gaussian component
            save_path (str): Path to save results

        Returns:
            dict: Result indicating if the Gaussian should be frozen
        """
        result, output = self.evaluate_gaussian(individual, i, save_path)
        if result == "freeze":
            individual.gaussians[i].freeze = True
        else:
            individual.gaussians[i].freeze = False
        return {"result": result, "output": output}
        
    def __save_output_to_file(self, data, log_path):
        """
        Save evaluation outputs to a JSON file.

        Args:
            data (list): Data to save
            log_path (str): Path to save the file
        """
        with open(os.path.join(log_path, f"outputs_{self.iteration}.json"), 'w') as f:
            json.dump({"data": data}, f)
        
    def update_matching(self, matching_result):
        """
        Update Elo ratings based on match results and save images.

        Args:
            matching_result (dict): Results from elo_matching

        Returns:
            dict: Updated match results
        """
        winner = matching_result["winner"]
        loser = matching_result["loser"]
        output = matching_result["output"]
        img1_base64 = matching_result["img1_base64"]
        img2_base64 = matching_result["img2_base64"]
        if winner is None:
            return {"winner": None, "loser": None, "output": output}
        elo_update(winner, loser)
        winner.matches.add(loser.id)
        loser.matches.add(winner.id)
        img_folder = os.path.join(self.save_path, "image_pairs")
        os.makedirs(img_folder, exist_ok=True)
        img1_path = os.path.join(img_folder, f"{self.iteration}_{output['img1']}_{output['img2']}.png")
        img2_path = os.path.join(img_folder, f"{self.iteration}_{output['img2']}_{output['img1']}.png")
        with open(img1_path, "wb") as f:
            import base64
            f.write(base64.b64decode(img1_base64))
        with open(img2_path, "wb") as f:
            f.write(base64.b64decode(img2_base64))
        return {"winner": winner.id, "loser": loser.id, "output": output}
    
    def par_elo_tournament(self, population, mode, save_path, num_workers=None, repeat_num=1):
        """
        Run a parallel Elo tournament with Swiss pairing system.

        Args:
            population (list): Population to compete
            mode (str): Evaluation mode
            save_path (str): Path to save results
            num_workers (int, optional): Number of parallel workers. Defaults to None
            repeat_num (int, optional): Number of times to repeat each comparison. Defaults to 1

        Returns:
            list: Population sorted by final Elo rating
        """
        if Test: 
            return population
        n = len(population)
        rounds = int(math.log2(n)) + 2

        random.shuffle(population)
        pairs = []
        for i in range(0, n, 2):
            if i+1 < n:
                for _ in range(repeat_num):
                    pairs.append((population[i], population[i+1], mode, save_path))
        all_data = []
        data, avg_time = self.__par_matching(tasks = pairs, desc = "Initial Round", num_workers=num_workers, repeat_num=repeat_num)
        all_data.extend(data)

        for _ in range(rounds-1):
            pairs = swiss_pairing(population)
            pairs_with_repeat = []
            for p1, p2 in pairs:
                for _ in range(repeat_num):
                    pairs_with_repeat.append((p1, p2, mode, save_path))
            pairs = pairs_with_repeat
            data, time = self.__par_matching(tasks = pairs, desc = "Swiss Pairing Round")
            all_data.extend(data)
            avg_time += time
        
        avg_time /= (rounds + 1)

        final_ranking = sorted(population, key=lambda x: (-x.rating))
        self.__save_output_to_file(all_data, save_path)
        return final_ranking
    
    def roulette_selection(self, population, num_parents):
        """
        Select parents using roulette wheel selection based on Elo ratings.

        Args:
            population (list): Population to select from
            num_parents (int): Number of parents to select

        Returns:
            list: Selected parents
        """
        min_rating = population[len(population) - 1].rating
        offset_ratings = [individual.rating - min_rating + 1 for individual in population]
        total_fitness = sum(offset_ratings)
        
        cum_probs = []
        cum = 0.0
        for i in range(len(population)):
            cum += offset_ratings[i] / total_fitness
            cum_probs.append(cum)
        
        selected_parents = []
        for _ in range(num_parents):
            r = random.random()
            idx = bisect.bisect_left(cum_probs, r)
            selected_parents.append(population[idx])
        return selected_parents
    
    def elo_process(self, population, mode, save_path=None, num_workers=None):
        """
        Run complete Elo rating process including tournament and image saving.

        Args:
            population (list): Population to process
            mode (str): Evaluation mode
            save_path (str, optional): Path to save results. Defaults to None
            num_workers (int, optional): Number of parallel workers. Defaults to None

        Returns:
            list: Population sorted by Elo rating
        """
        self.save_population_images(population=population, save_path=os.path.join(self.save_path, "population_state_nosort" + str(self.iteration) + ".png"))
        sorted_population = self.par_elo_tournament(population=population, mode=mode, save_path=self.save_path, num_workers=num_workers)
        sorted_population[0].render_image().save(os.path.join(self.save_path, "best_" + str(self.iteration) + ".png"))
        self.save_population_images(population=sorted_population, save_path=os.path.join(self.save_path, "population_state_" + str(self.iteration) + ".png"))
        return sorted_population
    
    def freeze_satisfied_features(self, population, save_path, num_workers=None):
        """
        Freeze satisfied Gaussian features across the population.

        Args:
            population (list): Population to process
            save_path (str): Path to save results
            num_workers (int, optional): Number of parallel workers. Defaults to None
        """
        self.par_population_freeze(population=population, save_path=save_path, num_workers=num_workers)
        self.save_activate_gasussian_images(population=population, save_path=os.path.join(self.save_path, "population_state_freezed" + str(self.iteration) + ".png"))
        
    def calculate_dynamic_fitness(self, population: List[TFparamsBase], current_gen, max_gen, min_pressure=1.2, max_pressure=4.0, k=2.0):
        """
        Calculate dynamic fitness values for the population based on current generation.

        Args:
            population (List[TFparamsBase]): Population to calculate fitness for
            current_gen (int): Current generation number
            max_gen (int): Maximum number of generations
            min_pressure (float, optional): Minimum selection pressure. Defaults to 1.2
            max_pressure (float, optional): Maximum selection pressure. Defaults to 4.0
            k (float, optional): Pressure scaling factor. Defaults to 2.0
        """
        progress = min(1.0, current_gen / max_gen)
        current_pressure = min_pressure + (max_pressure - min_pressure) * (progress ** k)
        
        n = len(population)
        for i in range(len(population)):
            rank = i + 1
            fitness = (n - rank + 1) ** current_pressure
            population[i].rating = fitness
    
    def roulette_and_elite(self, population, population_size, bound, generations, mode="quality", save_path=".", intent_interval = None, num_workers=None):
        """
        Perform selection combining roulette wheel and elitism.

        Args:
            population (list): Current population
            population_size (int): Target population size
            bound (Bound): Parameter boundaries
            generations (int): Total number of generations
            mode (str, optional): Evaluation mode. Defaults to "quality"
            save_path (str, optional): Path to save results. Defaults to "."
            intent_interval (int, optional): Interval for text-based evaluation. Defaults to None
            num_workers (int, optional): Number of parallel workers. Defaults to None

        Returns:
            list: New population after selection and mutation
        """
        self.calculate_dynamic_fitness(population, self.iteration, generations)
        elite_retention, random_generation = self.genetic_config.get_config(iter=self.iteration, maxiter=generations)
        elite_count = max(int(elite_retention * population_size), 1)
        random_count = 0
            
        elite_population = population[:elite_count]
        remaining_count = population_size - elite_count - random_count
        roulette_population = self.roulette_selection(population, remaining_count)
        
        elite_crossed_population = self.population_crossover(population=elite_population)
        roulette_crossed_population = self.population_crossover(population=roulette_population)
        crossed_population = elite_crossed_population + roulette_crossed_population
        
        mutated_population = self.population_mutation(population=crossed_population, bound=bound, iter=self.iteration - self.iter_bias, max_iter=generations - self.iter_bias,num_workers=num_workers)
        
        return mutated_population
        
    def eval(self, population, save_path=None, mode="quality"):
        """
        Evaluate and sort population using Elo tournament.

        Args:
            population (list): Population to evaluate
            save_path (str, optional): Path to save results. Defaults to None
            mode (str, optional): Evaluation mode. Defaults to "quality"

        Returns:
            list: Sorted population by Elo rating
        """
        sorted_population = self.par_elo_tournament(population, mode, save_path)
        
        sorted_population[0].render_image().save(os.path.join(save_path, "eval_best_" + str(self.iteration) + ".png"))
        self.save_population_images(population=sorted_population, save_path=os.path.join(save_path, "eval_population_state_" + str(self.iteration) + ".png"))
        return sorted_population
        
    def run_evaluation(
        self, 
        bound, 
        save_path='.',
        intent_interval=10
    ):
        """
        Run evaluation phase of the genetic algorithm.

        Args:
            bound (Bound): Parameter boundaries
            save_path (str, optional): Path to save results. Defaults to '.'
            intent_interval (int, optional): Interval for text-based evaluation. Defaults to 10
        """
        os.makedirs(save_path, exist_ok=True)
        population = self.population
        for i in range(len(self.population)):
            population[i].load_render_settings(bound=bound, volume=self.volume, gradient=self.gradient, step_size=self.setting.get_stepsize())
        self.parallel_render_population_image(population=population)
        
        for id, individual in enumerate(population):
            individual.reset_matching(id)
        if self.iteration >= intent_interval:
            mode = 'text'
        else:
            mode = 'quality'
            
        sorted_population = self.eval(population=population, save_path=save_path, mode=mode)
        state_path = os.path.join(self.save_path, "population_state_" + str(self.iteration) + ".pkl")
        GeneticAlgorithm.save_state(population=sorted_population, iteration=self.iteration, mode=mode, filename=state_path)
        
    def run_incremental(
        self, 
        bound, 
        mode="query",
        one_epoch=5,
        bg_color=None,
        save_path='.'
    ):
        """
        Run incremental evolution for a specified number of epochs.
        
        Args:
            bound (Bound): Parameter boundaries
            mode (str, optional): Evaluation mode. Defaults to "query"
            one_epoch (int, optional): Number of epochs to run. Defaults to 5
            bg_color (tuple, optional): Background color. Defaults to None
            save_path (str, optional): Path to save results. Defaults to '.'
        """
        os.makedirs(self.save_path, exist_ok=True)
        
        assert self.population is not None, "Population must be provided"
        
        if bg_color:
            self.bg_color = bg_color
            for i in range(len(self.population)):
                self.population[i].set_bg_color(bg_color)
        
        target_generation = self.iteration + one_epoch
        
        if self.iteration == 0:
            self.population = self.elo_process(population=self.population, mode=mode, save_path=self.save_path)
        
        for generation in tqdm(range(self.iteration, target_generation), desc="Running generation iteration:"):
            new_population = self.roulette_and_elite(population=self.population, population_size=len(self.population), bound=bound, generations=target_generation, mode=mode, save_path=self.save_path)
            
            self.population = new_population
            
            for i in range(len(self.population)):
                self.population[i].reset_matching(i)
                
            self.parallel_render_population_image(population=self.population)
            
            self.gaussian_of_population(self.population)
            self.gen_gif_from_images(output_gif_path=os.path.join(self.save_path, "tf_graph.gif"))
            
            self.population = self.elo_process(population=self.population, mode=mode, save_path=self.save_path)
            
            self.iteration += 1
            
        state_path = os.path.join(self.save_path, "population_state_" + str(self.iteration) + ".pkl")
        GeneticAlgorithm.save_state(population=self.population, iteration=self.iteration, mode=mode, filename=state_path)

    def run(
        self, 
        bound, 
        population_size=50, 
        generations=10, 
        save_path='.',
        intent_interval=10
    ):
        """
        Run the complete genetic algorithm.

        Args:
            bound (Bound): Parameter boundaries
            population_size (int, optional): Size of population. Defaults to 50
            generations (int, optional): Number of generations. Defaults to 10
            save_path (str, optional): Path to save results. Defaults to '.'
            intent_interval (int, optional): Interval for text-based evaluation. Defaults to 10

        Returns:
            list: Final population
        """
        os.makedirs(self.save_path, exist_ok=True)
        log_file = os.path.join(self.save_path, "timing_log.txt")
        mode = "quality"
        population = self.population
        atexit.register(GeneticAlgorithm.save_state, population, self.iteration, mode, os.path.join(self.save_path, 'population_state_autosave.pkl'))
        if self.iteration > intent_interval:
            reset_color = True
        else:
            reset_color = False
        
        for generation in tqdm(range(self.iteration, generations), desc="Running generation iteraton:"):
            timing_log = {}
            for i in range(len(population)):
                population[i].reset_matching(i)
            if self.iteration >= intent_interval:
                mode = 'text'
                if reset_color == False:
                    for i in range(len(population)):
                        population[i].random_color(bound)
                    reset_color = True
                self.iter_bias = intent_interval
                self.genetic_config.text_mode = True
                if self.llm_evaluator.middle_img is not None:
                    self.llm_evaluator.reset_format("image")
                    mode = 'image'
            else:
                mode = 'quality'
            
            render_time = time.time()
            self.parallel_render_population_image(population=population)
            timing_log["rendering"] = time.time() - render_time
            
            gauss_time = time.time()
            self.gaussian_of_population(population)
            self.gen_gif_from_images(output_gif_path=os.path.join(self.save_path, "tf_graph.gif"))
            timing_log["visualize_gaussians"] = time.time() - gauss_time
            
            elo_time = time.time()
            sorted_population = self.elo_process(population=population, mode=mode, save_path=self.save_path)
            timing_log["elo"] = time.time() - elo_time
            
            if self.iteration % self.save_interval == 0:
                start_time = time.time()
                state_path = os.path.join(self.save_path, "population_state_" + str(self.iteration) + ".pkl")
                GeneticAlgorithm.save_state(population=sorted_population, iteration=self.iteration, mode=mode, filename=state_path)
                timing_log["save"] = time.time() - start_time
                
            evolution_time = time.time()
            new_population=self.roulette_and_elite(population=sorted_population, population_size=population_size, bound=bound, generations=generations, mode=mode, save_path=self.save_path, intent_interval=intent_interval)
            timing_log["roulette_and_elite"] = time.time() - evolution_time
            
            population = new_population
            
            with open(log_file, "a") as f:
                f.write(f"\nGeneration {self.iteration} timing:\n")
                for stage, time_taken in timing_log.items():
                    f.write(f"  {stage}: {time_taken:.2f}s\n")
                total_time = sum(timing_log.values())
                f.write(f"  Total generation time: {total_time:.2f}s\n")
    
            self.iteration += 1
            
        for i in range(len(population)):
            population[i].reset_matching(i)
        self.parallel_render_population_image(population=population)
        sorted_population = self.eval(population=population, save_path=self.save_path, mode="text")
        self.gaussian_of_population(population)
        self.gen_gif_from_images(output_gif_path=os.path.join(self.save_path, "tf_graph.gif"))
        state_path = os.path.join(self.save_path, "population_state_" + str(self.iteration) + ".pkl")
        GeneticAlgorithm.save_state(population=sorted_population, iteration=self.iteration, mode=mode, filename=state_path)
        return population
    
    def __process_pairwise_cmp(self, task):
        """
        Process a single pairwise comparison task.

        Args:
            task (tuple): Tuple containing (player1, player2, mode, save_path)

        Returns:
            dict: Comparison results with timing information
        """
        starttime = time.time()
        player1, player2, mode, save_path = task
        result = self.elo_matching(player1, player2, mode, save_path)
        result["time"] = time.time() - starttime
        return result
    
    def __process_duplicate_pairs(self, cmp_results):
        """
        Process duplicate comparison results by taking the majority vote.
        
        Args:
            cmp_results (list): List of comparison results
            
        Returns:
            list: Deduplicated results with majority winners
        """
        pair_results = {}
        
        for result in cmp_results:
            if result["winner"] is None:
                continue
            
            key = tuple(sorted([result["winner"].id, result["loser"].id]))
            
            if key not in pair_results:
                pair_results[key] = []
            pair_results[key].append(result)
        
        unique_results = []
        for results in pair_results.values():
            winner_counts = {}
            for r in results:
                winner = r["winner"]
                winner_counts[winner.id] = winner_counts.get(winner.id, 0) + 1
            
            max_count = 0
            majority_result = None
            for r in results:
                count = winner_counts[r["winner"].id]
                if count > max_count:
                    max_count = count
                    majority_result = r
            
            unique_results.append({
                "winner": majority_result["winner"],
                "loser": majority_result["loser"],
                "output": majority_result["output"],
                "img1_base64": majority_result["img1_base64"],
                "img2_base64": majority_result["img2_base64"]
            })
        
        return unique_results
    
    def __par_matching(self, tasks, desc, num_workers=None, repeat_num=1):
        """
        Execute pairwise comparisons in parallel.

        Args:
            tasks (list): List of comparison tasks
            desc (str): Description for progress bar
            num_workers (int, optional): Number of parallel workers. Defaults to None
            repeat_num (int, optional): Number of times to repeat each comparison. Defaults to 1

        Returns:
            tuple: (results data, average time per task)
        """
        executor = ParallelExecutor(num_workers=num_workers)
        cmp_result = executor.execute(
            tasks = tasks,
            task_fn = self.__process_pairwise_cmp,
            desc=desc
            )
        if repeat_num > 1:
            cmp_result = self.__process_duplicate_pairs(cmp_result)
        avg_time = sum([result["time"] for result in cmp_result]) / len(cmp_result)
        data = []
        for pair in cmp_result:
            result = self.update_matching(pair)
            data.append(result)
        return data, avg_time

def parse_args(mode="train"):
    def parse_range(value_str):
        """Helper function to parse a single value or a range."""
        if ',' in value_str:
            return tuple(map(float, value_str.split(',')))  # 解析成元组
        else:
            return float(value_str)  # 单一数字
    
    def parse_int_list(value_str):
        """Helper function to parse a string into either a list of integers or a single value."""
        if ',' in value_str:
            return [int(x) for x in value_str.split(',')]
        else:
            return int(value_str)
        
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm with specified parameters.")

    # 添加命令行参数
    parser.add_argument('--config_file', type=str, required=True, help="Path to the bound config file.")
    parser.add_argument('--state_path', type=str, default=None, help="Previous state to be optimized on, default random.")
    parser.add_argument('--population_size', type=int, default=20, help="Size of the population.")
    parser.add_argument('--generations', type=int, default=10, help="Number of generations.")
    parser.add_argument('--save_interval', type=int, default=5, help="Interval of saving population state.")
    parser.add_argument('--cam_mutation_rate', type=str, default="0.05,0.4", help="Camera mutation rate, can be a single number or a range 'min,max'.")
    parser.add_argument('--cam_mutation_scale', type=str, default="0.05,0.2", help="Camera mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--op_mutation_rate', type=str, default="0.1,0.4", help="Opacity mutation rate, can be a single number or a range 'min,max'.")
    parser.add_argument('--op_mutation_scale', type=str, default="0.05,0.4", help="Opacity mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--x_mutation_scale', type=str, default="0.01,0.04", help="Gaussian x place mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--bandwidth_mutation_scale', type=str, default="0.05,0.2", help="Gaussian bandwidth place mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--color_mutation_rate', type=str, default="0.2,0.4", help="Color mutation rate, can be a single number or a range 'min,max'.")
    parser.add_argument('--H_mutation_scale', type=str, default="0.1,0.15", help="Color H mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--SL_mutation_scale', type=str, default="0.05,0.2", help="Color H mutation scale, can be a single number or a range 'min,max'.")
    parser.add_argument('--base_url', type=str, required=True, help="Base URL for the API.")
    parser.add_argument('--api_key', type=str, required=True, help="API key for authentication.")
    parser.add_argument('--prompt_folder', type=str, required=True, help="Folder where the prompt files are.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the state.")
    parser.add_argument('--instruct_path', type=str, default=None, help="Path to the instruct config file, default without text instruct.")
    parser.add_argument('--instruct_number', type=str, default=None, help="Instruct number or name to choose, default random.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run on (cuda / cpu), default is cuda.")
    parser.add_argument('--bg_color', type=str, default="(0, 0, 0)", help="Background color for the image.")
    parser.add_argument('--quality_metrics', type=str, default="16,11,14", help="The metrics used to evaluate rendered images' quality, default (16,11,14). Refer to ./prompt/aspects.json for more metrics info")
    parser.add_argument('--text_metrics', type=str, default="5", help="The metrics used to evaluate rendered images' alignment with text, default 5. Refer to ./prompt/aspects.json for more metrics info")
    parser.add_argument('--intent_interval', type=int, default=0, help="The round to start aligning with user's intent (whether image or text), default 0")
    parser.add_argument('--model_name', type=str, default="gpt-4o", help="Model name for the API.")
    parser.add_argument('--new_save_path', action='store_true', help="Whether to create a new save path.")
    parser.add_argument('--style_image', type=str, default=None, help="The styling image path.")
    parser.add_argument('--renderer', type=str, default="diffdvr", choices=["diffdvr", "anari"], help="Renderer to use (diffdvr or anari)")  # 添加渲染器选择参数
    if mode == "backend":
        parser.add_argument('--port', type=int, default=6006, help='服务器起始端口号 (默认: 6006)')
        parser.add_argument('--max_attempts', type=int, default=10, help='尝试查找可用端口的最大次数 (默认: 10)')
    
    # 解析命令行参数
    args = parser.parse_args()
    args.cam_mutation_rate = parse_range(args.cam_mutation_rate)
    args.cam_mutation_scale = parse_range(args.cam_mutation_scale)
    args.op_mutation_rate = parse_range(args.op_mutation_rate)
    args.op_mutation_scale = parse_range(args.op_mutation_scale)
    args.x_mutation_scale = parse_range(args.x_mutation_scale)
    args.bandwidth_mutation_scale = parse_range(args.bandwidth_mutation_scale)
    args.color_mutation_rate = parse_range(args.color_mutation_rate)
    args.H_mutation_scale = parse_range(args.H_mutation_scale)
    args.SL_mutation_scale = parse_range(args.SL_mutation_scale)
    
    print("args.bg_color: ", args.bg_color)
    
    # Parse background color from string to tuple
    args.bg_color = ast.literal_eval(args.bg_color)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    
    # Create config manager
    config_manager = ConfigManager(config_file=args.config_file, args=vars(args))
    
    # Setup API configuration
    config_manager.setup_api_config(
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_folder=args.prompt_folder
    )
    
    # Update algorithm configuration
    config_manager.update_algorithm_config(
        save_path=args.save_path
    )
    
    if args.state_path:
        population, iteration, mode = GeneticAlgorithm.load_state(args.state_path)
    else:
        population, iteration, mode = None, 0, []
        
    print("current iteration: ", iteration)
    print(f"Using renderer: {args.renderer}")  # 添加渲染器信息输出
    
    genetic_algorithm = GeneticAlgorithm(
        config_manager=config_manager,
        population=population,
        iteration=iteration,
        device=args.device,
        renderer_type=args.renderer  # 传入渲染器类型
    )
    
    # Run genetic algorithm
    best_solution = genetic_algorithm.run(
        bound=config_manager.get_bound(), 
        population_size=config_manager.get_algorithm_config().population_size, 
        generations=config_manager.get_algorithm_config().generations, 
        save_path=config_manager.get_algorithm_config().save_path,
        intent_interval=config_manager.get_algorithm_config().intent_interval
    )
    
