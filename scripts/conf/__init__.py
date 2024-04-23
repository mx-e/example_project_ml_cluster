import yaml
import os
import sys
import subprocess
import hydra
from pathlib import Path
from conf.base_config import SlurmConfig
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type, get_type_hints

import logging

def parse_dataclass(cls: Type, data: Dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data  # Not a data class, return the original data
    kwargs = {}
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name in data:
            field_data = data[f.name]
            field_type = type_hints[f.name]
            # Check if the field is a dataclass and field_data is a dictionary (nested dataclass)
            if is_dataclass(field_type) and isinstance(field_data, dict):
                kwargs[f.name] = parse_dataclass(field_type, field_data)
            else:
                kwargs[f.name] = field_data
    return cls(**kwargs)


def is_main(argv):
    return "--job-dir--" not in argv


def get_run_dir(argv: list[str]):
    return argv[argv.index("--job-dir--") + 1]


def reload_config(argv: list[str], config_class):
    with open(f"{get_run_dir(argv)}/.hydra/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg_parsed = parse_dataclass(config_class, cfg)  #
    if cfg_parsed.runtime_cfg:
        cfg_parsed.runtime_cfg.job_dir = get_run_dir(argv)
    return cfg_parsed


def create_slurm_run_script(
    prog_path: Path, container_path: Path, run_path: Path, slurm_conf: SlurmConfig
):
    os.mkdir(run_path / "logs")
    slurm_base_string = f"""#!/bin/bash
#SBATCH --job-name=auto_job
#SBATCH --partition={slurm_conf.partition}
#SBATCH --cpus-per-task={slurm_conf.cpus}
#SBATCH --output={run_path}/logs/job-%j.out
#SBATCH --error={run_path}/logs/job-%j.err
#SBATCH --mem={slurm_conf.mem}"""

    if slurm_conf.gpu_type:
        slurm_base_string += f"\n#SBATCH --gres=gpu:{slurm_conf.gpu_type}:1"

    if slurm_conf.notify_email:
        slurm_base_string += f"\n#SBATCH --mail-type=END,FAIL"
        slurm_base_string += f"\n#SBATCH --mail-user={slurm_conf.notify_email}"

    slurm_base_string += f"\n\napptainer run --nv {container_path} python {prog_path} --job-dir-- {run_path}"

    with open(run_path / "run.sh", "w") as f:
        f.write(slurm_base_string)
    return run_path / "run.sh"


def create_launch(conf):
    log_warning = None
    try:
        loglevel = getattr(logging, conf.loglevel.upper())
    except AttributeError:
        log_warning = f"Invalid log level {conf.loglevel}. Defaulting to INFO."
        loglevel = logging.INFO
    logging.basicConfig(level={loglevel}, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger(__name__)
    if log_warning: log.warning(log_warning)
    
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    main_file = os.path.abspath(sys.modules["__main__"].__file__)

    apptainer_path = Path.cwd() / conf.apptainer_path
    if not apptainer_path.exists():
        log.error(f"Apptainer container not found at {apptainer_path}. Exiting. Please configure a path to the run container.")
        sys.exit(1)
    ## create checkpoints dir
    os.makedirs(f"{run_dir}/ckpts", exist_ok=True)
    if conf.cluster:
        log.info(f"Starting run on cluster...")
        run_script_path = create_slurm_run_script(
            prog_path=main_file,
            container_path=apptainer_path,
            run_path=run_dir,
            slurm_conf=conf.slurm,
        )
        os.system(f"sbatch {run_script_path}")
    else:
        log.info(f"Starting run locally...")
        subprocess.run(
            [
                "apptainer",
                "run",
                "--nv",
                f"{apptainer_path}",
                "python",
                main_file,
                "--job-dir--",
                run_dir,
            ]
        )


def configure_run(config_class: Type):
    def wrapper(main_func):
        def wrapped_main(*args, **kwargs):
            conf = reload_config(sys.argv, config_class)
            main_func(conf, *args, **kwargs)

        return wrapped_main

    return wrapper
