# Example ML project setup on the IDA SLURM cluster using hydra for configuration and abstracting away Apptainer and SLURM
**Attention: This is not a generally recommended setup for an ML project on SLURM. Its rather a specialized solution for the IDA cluster.** 

## Motivation
For ML projects a good configuration management is important for documenting experiments and runs and simplifying parameter overrides. Type checking makes console overrides less error-prone. Multirun support simplifies hyperparameter tuning by allowing scheduling and launching of several jobs using a simple override syntax.

The [Hydra](https://hydra.cc) python package brings all of these features together with plugin support for custom sweepers and launchers. We therefore want to use Hydra for configuration management wherever possible. The SLURM launcher plugin for hydra even allows launching jobs on SLURM automatically when using multirun.


**There is, however, a problem with the standard setup on the SLRUM cluster of the IDA lab at TU Berlin:**

We use apptainer for isolation of experimental python environments. This makes python environment management cleaner and results more reproducible while also reducing the load on the shared cluster filesystem by storing environments in a single container file. However, it is not possible to use the slurm CLI commands of the cluster inside apptainer containers, which is necessary for launching jobs using the SLURM launcher plugin.


**We therefore need to use a workaround:**

We want to run a python hydra script running python outside apptainer which then launches the apptainer jobs on the cluster. To make this setup less complicated we "hack" hydra a little so adding this functionality to any ML experiments just takes a few extra lines of code. There are also additional benefits to this approach such as configuring the SLURM commands through hydra.


## Features
- **configure your experiments using hydra (inclding type checking, experiment overrides, command line overrides)**
- **configure SLURM parameters using hydra as well**
- **completely abstracts away the cluster and apptainer - you do not need SLURM sh files and can simply launch your python scripts**
- **jobs can be launched in local shell or submitted to SLURM by adding a single flag**
- **use hydra mutlirun to queue several SLURM jobs simultaneously**


## How to use

### Building container
Build the container using apptainer:

```apptainer build --nv container.sif container.def```

The 'nv' flag is important to build the CUDA backend of torch when a GPU is available - you might have to rebuild the container when switching from a CPU to a GPU node.

*You need to be on a SLURM node (srun shell)for this command to work.*


### Installing & removing dependencies
This example uses poetry for dependency management. To add and remove dependencies use 

```apptainer run --nv container.sif poetry --no-cache add --lock <dependencies>```

and 

```apptainer run --nv container.sif poetry --no-cache remove --lock <dependencies>```.

This will only update the pyproject.toml with the correct package and version information without touching the cluster filesystem otherwise or installing anything. Rebuild the container to use resolved packages.

### Launching scripts (on the cluster)
### Configuring SLURM



