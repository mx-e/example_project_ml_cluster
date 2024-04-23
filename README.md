# Example ML project setup on the IDA SLURM cluster using Hydra for configuration and abstracting away Apptainer and SLURM
**Attention: This is not a generally recommended setup for an ML project on SLURM. It's rather a specialized solution for the IDA cluster.** 

## Motivation
For ML projects, good configuration management is important for documenting experiments and runs and simplifying parameter overrides. Type checking makes console overrides less error-prone. Multirun support simplifies hyperparameter tuning by allowing scheduling and launching of several jobs using a simple override syntax.

The [Hydra](https://hydra.cc) Python package brings all of these features together with plugin support for custom sweepers and launchers. We therefore want to use Hydra for configuration management wherever possible. The SLURM launcher plugin for Hydra even allows launching jobs on SLURM automatically when using multirun.


**There is, however, a problem with the standard setup on the SLURM cluster of the IDA lab at TU Berlin:**

We use Apptainer for isolation of experimental Python environments. This makes Python environment management cleaner and results more reproducible while also reducing the load on the shared cluster filesystem by storing environments in a single container file. However, it is not possible to use the SLURM CLI commands of the cluster inside Apptainer containers, which is necessary for launching jobs using the SLURM launcher plugin.


**We therefore need to use a workaround:**

We want to run a Python Hydra script running Python outside Apptainer which then launches the Apptainer jobs on the cluster. To make this setup less complicated we "hack" Hydra a little so adding this functionality to any ML experiments just takes a few extra lines of code. There are also additional benefits to this approach such as configuring the SLURM commands through Hydra.


## Features
- **Configure your experiments using Hydra (including type checking, experiment overrides, command line overrides)**
- **Configure SLURM parameters using Hydra as well**
- **Completely abstracts away the cluster and Apptainer - you do not need SLURM sh files and can simply launch your Python scripts**
- **Jobs can be launched in local shell or submitted to SLURM by adding a single flag**
- **Use Hydra multirun to queue several SLURM jobs simultaneously**


## How to use
For information on configuration management and overrides see the [Hydra](https://hydra.cc) documentation.
Make sure to install Hydra core and PyYAML before using this setup:

```pip install hydra-core PyYAML```

### Building container
Build the container using Apptainer:

```apptainer build --nv container.sif container.def```

The 'nv' flag is important to build the CUDA backend of Torch when a GPU is available - you might have to rebuild the container when switching from a CPU to a GPU node.

*You need to be on a SLURM node (srun shell) for this command to work.*


### Installing & removing dependencies
This example uses poetry for dependency management. To add and remove dependencies use 

```apptainer run --nv container.sif poetry --no-cache add --lock <dependencies>```

and 

```apptainer run --nv container.sif poetry --no-cache remove --lock <dependencies>```.

This will only update the pyproject.toml with the correct package and version information without touching the cluster filesystem otherwise or installing anything. Rebuild the container to use resolved packages.

### Launching scripts
Run scripts simply through Python:

```python example_script_1.py```

To override single parameters, use Hydra's override syntax (see [Hydra docs](https://hydra.cc)):

```python example_script_1.py training.lr=0.01```

To make multiple overrides - for instance to configure an experiment, you can create a YAML file specifying the overrides. An example is provided under /scripts/conf/experiment. Use it with:

```python example_script_1.py +experiment=ex1```

To start multiple runs simultaneously you can define multiple values for overrides. Include the multirun flag '-m':

```python example_script_1.py -m training.lr=0.1,0.01,0.001```

Experiments will run in the local shell by default. Multirun jobs will run in sequence. To submit a job or multiple jobs to the SLURM queue set the 'cluster' flag to 'True':

```python example_script_1.py -m training.lr=0.1,0.01,0.001 cluster=True```

You can mix overrides: 

```python example_script_1.py -m +experiment=ex1 training.lr=0.1,0.01,0.001 cluster=True model.decoder_max_length=22```

### Output directories
Hydra will create an out directory to which your SLURM logs will get saved as well. You can configure the path by editing the hydra_conf object in the config definition (script_x_config.py) (see [Hydra docs](https://hydra.cc)).


### Configuring SLURM
Our Hydra workaround also configures SLURM and automatically creates SLURM sh files and launches them. A 'run.sh' file will be created in the run's output directory when a job is submitted to the cluster. SLURM also gets configured using Hydra so you can override the SLURM config just like any other part of the config:

```python example_script_1.py +experiment=ex1 slurm.partition=gpu-5h cluster=True```

To change the default run config edit the SlurmConfig dataclass in ```/conf/base_conf.py```:

```
@dataclass
class BaseConfig:
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    cluster: bool = False
    apptainer_path: str = "../container.sif"
    loglevel: str = "info"
    runtime_cfg: RuntimeConfig = field(default_factory=RuntimeConfig)
```

**When creating configs for new run scripts make sure to inherit from the BaseConfig dataclass! This is necessary for the cluster flag, setting the loglevel, and loading the SLURM config.**

Our workaround needs to know the location of the current Apptainer container to work properly. If you move or rename the .sif file, edit the BaseConfig dataclass in ```/conf/base_conf.py```










