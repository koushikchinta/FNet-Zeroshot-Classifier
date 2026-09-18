import torch
import yaml
from dataclasses import dataclass
import logging


@dataclass
class ModelConfig:
    num_encoding_layers: int = 3
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    initial_checkpoint: str | None = None
    log: bool = True
    epochs: int = 10
    train_batch_size: int = 32
    validation_batch_size: int = 32
    base_lr: float = 3e-4
    weight_decay: float = 1e-5


@dataclass
class GlobalConfig:
    verbose: bool
    model: ModelConfig
    train: TrainingConfig
    logger: logging.Logger
    device: torch.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("config.yaml", "r") as f:
    _config_dict = yaml.safe_load(f)


_model_config = ModelConfig(**_config_dict.get("model", {}))
_training_config = TrainingConfig(**_config_dict.get("train", {}))
Config = GlobalConfig(model=_model_config, train=_training_config, **_config_dict)
