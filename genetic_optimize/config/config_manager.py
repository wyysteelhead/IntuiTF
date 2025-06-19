import json
import os
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from genetic_optimize.states.bound import Bound

@dataclass
class AlgorithmConfig:
    population_size: int = 50
    generations: int = 10
    save_interval: int = 5
    text_interval: int = 10
    device: str = "cuda"
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    model_name: str = "gpt-4o"
    save_path: str = "results"

@dataclass
class MutationConfig:
    cam_mutation_rate: Union[float, tuple] = (0.05, 0.4)
    cam_mutation_scale: Union[float, tuple] = (0.05, 0.2)
    op_mutation_rate: Union[float, tuple] = (0.1, 0.4)
    op_mutation_scale: Union[float, tuple] = (0.05, 0.4)
    x_mutation_scale: Union[float, tuple] = (0.01, 0.04)
    bandwidth_mutation_scale: Union[float, tuple] = (0.05, 0.2)
    color_mutation_rate: Union[float, tuple] = (0.2, 0.4)
    H_mutation_scale: Union[float, tuple] = (0.1, 0.15)
    SL_mutation_scale: Union[float, tuple] = (0.05, 0.2)

@dataclass
class APIConfig:
    base_url: str = ""
    api_key: str = ""
    prompt_folder: str = ""
    instruct_path: Optional[str] = None
    instruct_number: Optional[str] = None
    quality_metrics: str = "7"
    text_metrics: str = "5"

class ConfigManager:
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default_config.json')

    def __init__(self, config_file: str, custom_config_path: Optional[str] = None, args = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str): Path to the bound configuration file
            custom_config_path (str, optional): Path to a custom configuration file. If not provided,
                                              will try to load from default location or use default values.
        """
        self.config_file = config_file
        self.bound = None
        self.algorithm_config = AlgorithmConfig()
        self.mutation_config = MutationConfig()
        self.api_config = APIConfig()
        
        # Load configurations in order: defaults -> custom file -> runtime updates
        self._load_default_config()
        if custom_config_path and os.path.exists(custom_config_path):
            self._load_custom_config(custom_config_path)
        
        # if args has not None parameters, update the config
        if args is not None:
            self.update_algorithm_config(**args)
        
        self.load_config()

    def _load_default_config(self) -> None:
        """Load default configuration from default_config.json or use dataclass defaults."""
        try:
            if os.path.exists(self.DEFAULT_CONFIG_PATH):
                with open(self.DEFAULT_CONFIG_PATH, 'r') as f:
                    config_data = json.load(f)
                    
                    # Update algorithm config
                    if 'algorithm' in config_data:
                        for key, value in config_data['algorithm'].items():
                            if key == 'bg_color' and isinstance(value, list):
                                value = tuple(value)  # Convert list to tuple for bg_color
                            setattr(self.algorithm_config, key, value)
                    
                    # Update mutation config
                    if 'mutation' in config_data:
                        for key, value in config_data['mutation'].items():
                            if isinstance(value, list):
                                value = tuple(value)  # Convert lists to tuples for mutation ranges
                            setattr(self.mutation_config, key, value)
                    
                    # Update API config
                    if 'api' in config_data:
                        for key, value in config_data['api'].items():
                            setattr(self.api_config, key, value)
        except Exception as e:
            print(f"Warning: Could not load default config file: {e}")
            print("Using dataclass defaults instead.")

    def _load_custom_config(self, config_path: str) -> None:
        """Load configuration from a custom config file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
                # Update configs with custom values
                if 'algorithm' in config_data:
                    self.update_algorithm_config(**config_data['algorithm'])
                if 'mutation' in config_data:
                    self.update_mutation_config(**config_data['mutation'])
                if 'api' in config_data:
                    self.setup_api_config(**config_data['api'])
        except Exception as e:
            raise ValueError(f"Error loading custom config from {config_path}: {e}")

    def load_config(self) -> None:
        """Load bound configuration from file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        self.bound = Bound(self.config_file)

    def update_algorithm_config(self, **kwargs) -> None:
        """Update algorithm configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.algorithm_config, key):
                if key == 'bg_color' and isinstance(value, list):
                    value = tuple(value)
                setattr(self.algorithm_config, key, value)
            # else:
                # raise ValueError(f"Invalid algorithm config parameter: {key}")

    def update_mutation_config(self, **kwargs) -> None:
        """Update mutation configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.mutation_config, key):
                if isinstance(value, list):
                    value = tuple(value)
                setattr(self.mutation_config, key, value)
            else:
                raise ValueError(f"Invalid mutation config parameter: {key}")

    def setup_api_config(self, base_url: str = "", api_key: str = "", prompt_folder: str = "", args = None, **kwargs) -> None:
        """Setup API configuration."""
        self.api_config = APIConfig(
            base_url=base_url,
            api_key=api_key,
            prompt_folder=prompt_folder,
            **kwargs
        )

    def get_bound(self) -> Bound:
        """Get the bound object."""
        return self.bound

    def get_algorithm_config(self) -> AlgorithmConfig:
        """Get algorithm configuration."""
        return self.algorithm_config

    def get_mutation_config(self) -> MutationConfig:
        """Get mutation configuration."""
        return self.mutation_config

    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return self.api_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "algorithm": asdict(self.algorithm_config),
            "mutation": asdict(self.mutation_config),
            "api": asdict(self.api_config)
        }

    def save_config(self, save_path: str) -> None:
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(cls, bound_config: str, config_path: str) -> 'ConfigManager':
        """Create a ConfigManager instance from a configuration file."""
        return cls(bound_config, custom_config_path=config_path) 