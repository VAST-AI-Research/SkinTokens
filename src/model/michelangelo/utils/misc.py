# -*- coding: utf-8 -*-

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig
import time
import torch
import torch.distributed as dist
from typing import Union, Any, Optional
from collections import defaultdict
from torch.optim import lr_scheduler
import os
from dataclasses import dataclass, field
from contextlib import contextmanager

import logging
logger = logging.getLogger(__name__)



def calc_num_train_steps(num_data, batch_size, max_epochs, num_nodes, num_cards=8):
    return int(num_data / (num_nodes * num_cards * batch_size)) * max_epochs


OmegaConf.register_new_resolver("calc_num_train_steps", calc_num_train_steps)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

@dataclass
class ExperimentConfig:
    task: str = "vae"
    output_dir: str = "outputs"
    resume: Optional[str] = None

    data: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    
    trainer: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)

    wandb: dict = field(default_factory=dict)

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg

def get_config_from_file(config_file: str, cli_args: list = [], **kwargs) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)
    cli_conf = OmegaConf.from_cli(cli_args)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file.items() if key != "base_config"}

        cfg = OmegaConf.merge(base_config, config_file, cli_conf, kwargs)
    else:
        cfg = OmegaConf.merge(config_file, cli_conf, kwargs)
    
    scfg: ExperimentConfig = parse_structured(ExperimentConfig, cfg)

    return scfg

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_free_space(path):
    fs_stats = os.statvfs(path)
    free_space = fs_stats.f_bsize * fs_stats.f_bfree
    return free_space

def get_device_type():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)

def get_hostname():
    import socket
    return socket.gethostname()

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
    
def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError

def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler

class TimeRecorder:
    _instance = None

    def __init__(self):
        self.items = {}
        self.accumulations = defaultdict(list)
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
        self.enabled = False

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = super(TimeRecorder, cls).__new__(cls)
        return cls._instance

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str, accumulate: bool = False) -> float:
        if not self.enabled or name not in self.items:
            return
        torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        if accumulate:
            self.accumulations[name].append(delta)
        t = delta * self.time_scale
        logger.info(f"{name}: {t:.2f}{self.time_unit}")

    def get_accumulation(self, name: str, average: bool = False) -> float:
        if not self.enabled or name not in self.accumulations:
            return
        acc = self.accumulations.pop(name)
        total = sum(acc)
        if average:
            t = total / len(acc) * self.time_scale
        else:
            t = total * self.time_scale
        logger.info(f"{name} for {len(acc)} times: {t:.2f}{self.time_unit}")


### global time recorder
time_recorder = TimeRecorder()

class FLASH3:
    def __init__(self) -> None:
        self.available = "H100" in get_device_type()
        self.use = os.environ.get("USE_FLASH3", False)
    
    @property
    def is_use(self):
        return self.available and self.use
    
    @contextmanager
    def disable_flash3(self):
        use = self.use
        self.set_use(False)
        yield
        self.set_use(use)

    def set_use(self, use=True):
        self.use = use

use_flash3 = FLASH3()
