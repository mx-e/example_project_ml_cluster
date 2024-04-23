from dataclasses import dataclass, field

from typing import Any
from hydra.core.config_store import ConfigStore
from conf.base_config import BaseConfig


## EDIT OUT DIR PATH HERE 
OUT_SUBPATH = "script_1"

hydra_conf =  {
    "run": {"dir": f"../out/{OUT_SUBPATH}"+"/${now:%Y-%m-%d_%H-%M-%S}"},
    "sweep": {
        "dir": f"../out/{OUT_SUBPATH}"+"/sweep/${now:%Y-%m-%d_%H-%M-%S}",
    },
    "mode": "RunMode.RUN",
}
##
 
cs = ConfigStore.instance()

@dataclass
class TrainingConfig:
    batch_size: int = 64
    lr: float = 1e-04
    lr_anneal_steps: int = 200000
    warmup_steps: int = 10000
    ema_rate: float = 0.9999
    weight_decay: float = 0.0
    gradient_clipping: float = -1.0
    eval_interval: int = 2000
    save_interval: int = 50000
    seed: int = 1108


@dataclass
class ModelConfig:
    encoder_max_length: int = 64
    decoder_max_length: int = 64


@dataclass
class DataConfig:
    data_name: str
    data_path: str


@dataclass
class YelpDataConfig(DataConfig):
    data_name: str = "yelp"
    data_path: str = "data/yelp"

cs.store(group="data", name="yelp", node=YelpDataConfig)

@dataclass
class IMDBDataConfig(DataConfig):
    data_name: str = "imdb"
    data_path: str = "data/imdb"
    labels: list = field(default_factory=lambda: ["pos", "neg"])

cs.store(group="data", name="imdb", node=IMDBDataConfig)

defaults = [
    {"data": "yelp"},
    "_self_",
]

@dataclass
class Script1Config(BaseConfig):
    data: Any = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: bool = False
    hydra: dict = field(default_factory=lambda: hydra_conf)
    defaults: list = field(default_factory=lambda: defaults)

cs.store(name="conf", node=Script1Config)

