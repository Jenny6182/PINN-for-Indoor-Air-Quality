"""
Base pipeline class defining the common interface for all model types.
Each pipeline (simple, varying, raapinn, one_raapinn) inherits from this.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import yaml


class BasePipeline(ABC):
    """Abstract base class for PINN pipelines."""
    
    def __init__(self, config_path: str = None, run_dir: Path = None):
        """
        Parameters
        ----------
        config_path : str
            Path to YAML config file. If None, uses defaults.
        run_dir : Path
            Output directory. If None, creates one based on model type.
        """
        self.config = self._load_config(config_path)
        self.run_dir = run_dir or self._create_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path):
        """Load config from YAML file or return defaults."""
        if config_path:
            with open(config_path) as f:
                return yaml.safe_load(f)
        return self.get_default_config()
    
    def _create_run_dir(self):
        """Create output directory based on model type and timestamp."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"results/{self.model_type}/{timestamp}")
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier (e.g., 'simple', 'raapinn')."""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> dict:
        """Return default configuration dictionary."""
        pass
    
    @abstractmethod
    def run(self, data_path: str, **kwargs) -> dict:
        """
        Run the pipeline.
        
        Parameters
        ----------
        data_path : str
            Path to CSV data file
        **kwargs
            Model-specific parameters
            
        Returns
        -------
        dict
            Results dictionary with 'taus', 'Q_minus', 'Q_plus', etc.
        """
        pass
    
    def save_config(self):
        """Save config to run_dir for reproducibility."""
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_type='{self.model_type}')"
