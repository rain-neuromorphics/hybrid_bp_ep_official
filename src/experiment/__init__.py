from omegaconf import DictConfig
import abc
from torch.multiprocessing import Process
from typing import Optional
import torch
import numpy as np
from hydra.utils import get_original_cwd
import os

class ExperimentMeta(type):
    """
    Metaclass to register Experiment classes inside a dictionary mapping
    experiment type name to associated experiment class:
    
    ExperimentMeta.REGISTRY = {"gdd": GDDExperiment, "training": TrainingExperiment}    
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(ExperimentMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

class Experiment(metaclass=ExperimentMeta):
    """
    Abstract class to build an experiment
    
    Attributes
    ---------
    cfg (DictConfig):
        hydra configuration
    device (torch.device):
        device at use for the experiment
    multiprocessing (bool):
        specifies whether we use data parallelism
    load_dir (str):
        directory where we load the model from (if any)
    save_dir (str):
        directory where we save the results into
    save (bool):
        specifies whether we save the model
    
    Methods
    -------
    _run :
        runs the experiments
    run:
        wrapper around _run
    """
    
    def __init__(
        self, 
        cfg: DictConfig,
        device: int = 0,
        seed: Optional[int] = None,
        load_dir: Optional[str] = None,
        save: bool = False,
        multiprocessing: bool = False,
        **kwargs
        ) -> None:

        self.cfg = cfg
        self.device = device
        self.multiprocessing = multiprocessing
        
        self.load_dir = load_dir if not load_dir else get_original_cwd() + '/outputs/' + load_dir + '/checkpoint.pth' # load_dir = date / run_id, as automatically formatted by hydra
        self.save_dir = os.getcwd()
        self.save = save
        
        if seed:
            torch.manual_seed("{}".format(seed))
            np.random.seed(seed)

    def run(self) -> None:
        self._run()

    @abc.abstractmethod
    def _run(self, rank, world_size):
        raise NotImplementedError('_run method not implemented!')

from . import gdd, training