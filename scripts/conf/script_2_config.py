from __future__ import annotations

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from conf.base_config import BaseConfig

OUT_SUBPATH = "script_2"

hydra_conf =  {
    "run": {"dir": f"../out/{OUT_SUBPATH}"+"/${now:%Y-%m-%d_%H-%M-%S}"},
    "sweep": {
        "dir": f"../out/{OUT_SUBPATH}"+"/sweep/${now:%Y-%m-%d_%H-%M-%S}",
    },
    "mode": "RunMode.RUN",
}

cs = ConfigStore.instance()

@dataclass
class InferenceConfig:
    batch_size: int = 64
    seed: int = 1108

class ModelConfig:
    load_path: str = "model.pt"
    train_config_dir: str = "../out/script_1/2021-09-29_15-00-00"

@dataclass
class Script2Config(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    debug: bool = False
    hydra: dict = field(default_factory=lambda: hydra_conf)

cs.store(name="conf", node=Script2Config)

