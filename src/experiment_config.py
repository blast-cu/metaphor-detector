from dataclasses import dataclass
from typing import Optional
import logging
import os
import json
import yaml

"""
Configs to capture all parameters for experiments implemented in 'experiments'. These
may be saved in json format to the output files.
"""

@dataclass(kw_only=True)
class ExperimentConfig:

    def __init__(
        self, task_type: str, task_name: str, dataset: str,
        logger: logging.Logger, env_vars: dict, sample_size: int = None, skip_load: bool = False, labeled_only: bool = False, 
        device: str = "auto", random_seed: int = 42, num_workers: int = 8, testing: bool = False
    ):
        self.task_type = task_type
        self.task_name = task_name
        self.dataset = dataset

        self.logger = logger
        self.env_vars = env_vars
        self.repo_dir = env_vars["REPO_PATH"]
        self.raw_data_dir = env_vars["RAW_DATA_DIR"]
        self.db_dir = env_vars["DB_DIR"]
        self.results_dir = env_vars["RESULTS_DIR"]

        self.skip_load = skip_load
        self.sample_size = sample_size
        self.labeled_only = labeled_only
        self.device = device
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.testing = testing

        self.dataset_out_name = self.dataset
        if self.labeled_only:
            self.dataset_out_name += "_labeled"

    @property
    def task_dir(self) -> str:
        """Get the experiment directory path."""
        task_dir = f"{self.results_dir}/{self.task_type}/{self.task_name}"
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
    
    @property
    def cache_dir(self) -> str:
        """Get the experiment directory path."""
        cache_dir = f"{self.repo_dir}/cache/{self.task_type}/{self.task_name}"
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @property
    def db_path(self) -> str:
        """Get the path to the database file."""
        os.makedirs(self.db_dir, exist_ok=True)
        db_path = f"{self.db_dir}/{self.dataset.lower()}.db"
        if self.labeled_only:
            db_path = db_path.replace(".db", f"_hand_labeled.db")
        return db_path
    
    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger, env_vars: dict) -> 'ExperimentConfig':
        """Create an ExperimentConfig from a dictionary."""
        return cls(
            task_type=data['task_type'], 
            task_name=data['task_name'],
            dataset=data['dataset'],
            logger=logger,
            env_vars = env_vars,
            skip_load = data["skip_load"],
            labeled_only=data.get('labeled_only', False),
            device=data.get('device', 'auto'),
            random_seed=data.get('random_seed', 42)
        )

    
    def to_dict(self) -> dict:
        """Convert the config to a dictionary."""
        return {
            "task_type": self.task_type,
            "task_name": self.task_name,
            "dataset": self.dataset,
            "repo_dir": self.repo_dir,
            "raw_data_dir": self.raw_data_dir,
            "db_dir": self.db_dir,
            "db_path": self.db_path,
            "skip_load": self.skip_load,
            "labeled_only": self.labeled_only,
            "device": self.device,
            "random_seed": self.random_seed
        }
    
    def save_config(self):
        """Save the config to a file in the experiment folde."""
        task_dir = self.task_dir
        out_config = self.to_dict()
        with open(f"{task_dir}/config.json", "w") as f:
            json.dump(out_config, f, indent=4)
        self.logger.info(f"Config saved to {task_dir}/config.json")


@dataclass(kw_only=True)
class LLMExperimentConfig(ExperimentConfig):
    """Configuration for metaphor filtering experiments (e.g. filtering out non-metaphoric documents)."""

    def __init__(
        self, dataset: str, task_type: str, prompt_file: str,
        logger: logging.Logger, env_vars: dict, model_config_file: str, host: str, port: int,
        sample_size: Optional[int] = None, testing: bool = False, skip_load: bool = False, labeled_only: bool = False, 
        device: str = "auto", random_seed: int = 42, shots: int = None, out_file = None, num_workers: int = 8
    ):
        task_name = "llm"  # this is the type of the task
        super().__init__(
            task_type=task_type, task_name=task_name, dataset=dataset,
            logger=logger, env_vars=env_vars, skip_load=skip_load, labeled_only=labeled_only, 
            device=device, random_seed=random_seed, num_workers=num_workers,
            testing=testing
        )
        self.prompt_file = prompt_file
        self.model_config_file = model_config_file
        self.load_model_config()
        self.answer_format = None
        self.host = host
        self.port = port
        self.sample_size = sample_size
        self.shots = shots
        self.out_file = out_file
    
    def check_label_setting(self):
        # verify that hand_labeled data is only used with tweets dataset
        if self.labeled_only and self.dataset != 'tweets_immigration':
            raise ValueError("The 'labeled_only' option is only supported for the 'tweets_immigration' dataset.")

    def load_model_config(self):
        """Load model configuration from a JSON file."""
        config_path = os.path.join(self.repo_dir, "src/llm_ann/configs", self.model_config_file)
        self.logger.info(f"Loading annotation config from {config_path}")
        temp_config = yaml.safe_load(open(config_path)) if self.model_config_file else {}
        self.model = temp_config["model"]
        self.temperature = temp_config.get("temperature", 1.0)
        self.workers = temp_config.get("workers", 1)
        self.save_interval = temp_config.get("save_interval", 500)
        self.max_retries = temp_config.get("max_retries", 5)

    @property
    def out_path(self) -> str:
        """Get the output path for the experiment results."""
        out_filename = f"{self.task_dir}/{self.dataset}"
        if self.labeled_only:
            out_filename += "_hand_labeled"
        if self.testing:
            out_filename += "_testing"
        if self.sample_size is not None:
            out_filename += f"_sampled_{self.sample_size}"
        out_filename += ".json"
        return out_filename

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger, env_vars) -> 'LLMExperimentConfig':
        """Create a LLMExperimentConfig from a dictionary."""
        # reuse the parent class method
        return cls(
            env_vars = env_vars,
            logger=logger,
            dataset=data['dataset'],
            prompt_file=data['prompt_file'],
            model_config_file=data.get('model_config_file'),
            labeled_only=data['labeled_only'],
            device=data["device"],
            random_seed=data["random_seed"],
            task_type=data["task_type"],
            host=data["host"],
            port=data["port"]
        )
        
    def to_dict(self) -> dict:  
        """Convert the config to a dictionary."""
        out_config = super().to_dict()
        out_config.update({
            "prompt_file": self.prompt_file,
            "model_config_file": self.model_config_file,
            "model": self.model,
            "temperature": self.temperature,
            "workers": self.workers,
            "save_interval": self.save_interval,
            "max_retries": self.max_retries,
            "host": self.host,
            "port": self.port,
            "testing": self.testing,
            "sample_size": self.sample_size
        })
        return out_config