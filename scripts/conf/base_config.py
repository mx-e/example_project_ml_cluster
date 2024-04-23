from __future__ import annotations

from typing import Union
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class SlurmConfig:
    partition: str = "cpu-5h"
    cpus: int = 4
    gpu_type: Union[str, None] = None
    output_dir: str = "logs"
    mem: str = "16GB"
    notify_email: Union[str, None] = None

@dataclass
class RuntimeConfig:
    device: Union[str, None] = MISSING
    n_gpu: Union[int, None] = MISSING
    job_dir: Union[str, None] = MISSING

@dataclass
class BaseConfig:
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    cluster: bool = False
    apptainer_path: str = "../container.sif"
    loglevel: str = "info"
    runtime_cfg: RuntimeConfig = field(default_factory=RuntimeConfig)


