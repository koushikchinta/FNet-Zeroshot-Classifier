import torch
import yaml
from dataclasses import dataclass
import logging
from logger import configure_logging


@dataclass
class ModelConfig:
    num_encoding_layers: int = 5
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    checkpoint_dir: str
    initial_checkpoint: str | None = None
    epochs: int = 10
    train_batch_size: int = 32
    validation_batch_size: int = 32
    base_lr: float = 3e-4
    weight_decay: float = 1e-5

@dataclass
class TestingConfig:
    run_tests: bool = True
    batch_size:int =  32


@dataclass
class GlobalConfig:
    model: ModelConfig
    train: TrainingConfig
    test: TestingConfig
    logger: logging.Logger
    log_level: str = "INFO"
    log_to_terminal: bool = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("config.yaml", "r") as f:
    _config_dict = yaml.safe_load(f)


_config_dict["model"] = ModelConfig(**_config_dict.get("model", {}))
_config_dict["train"] = TrainingConfig(**_config_dict.get("train", {}))
_config_dict["test"] = TestingConfig(**_config_dict.get('test', {}))
_config_dict["logger"] = configure_logging(
    _config_dict.get("log_level", "INFO"), _config_dict.get("log_to_terminal", True)
)
Config = GlobalConfig(**_config_dict)
