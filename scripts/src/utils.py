import torch as th


def set_runtime_cfg(cfg: any):
    assert hasattr(
        cfg, "runtime_cfg"
    ), "runtime_config is not defined for the configuration class"
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    n_gpu = th.cuda.device_count()
    cfg.runtime_cfg.device = device
    cfg.runtime_cfg.n_gpu = n_gpu
