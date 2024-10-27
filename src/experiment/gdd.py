from src.models import getModel
from src.helpers import copy
from src.algorithms import getAlgorithm, Algorithm
from .utils import getCriterion
from src.data.utils import getDataLoaders
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from torch import Tensor
from . import Experiment
from omegaconf import DictConfig

class GDDExperiment(Experiment):
    """
    Class implementing the static gradient analysis and checking the "Gradient Descending Dynamics" (GDD) property on ff-EBMs
    
    Attributes
    ----------
    plotter (str):
        kind of plotter at use
        
    Methods
    -------
    _run:
        same as parent class (Experiment)
    
    compute_gradients:
        computes gradients and detailed gradients * with respect to parameters and neurons * and * for a given algorithm *
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        plotter: str = "default",
        **kwargs
        ) -> None:
        """
        Instantiates a GDDExperiment object
        
        Args:
            cfg (DictConfig):
                configuration passed by yaml configuration and terminal prompt overrides
            plotter (str):
                type of plotter at use (use "paperplotter" to reproduce plots of our paper)
        """
        
        super().__init__(cfg, **kwargs)
        self.plotter = plotter
        
    def _run(self) -> None:
        """"
        Runs the static gradient analysis
        
        Args:
            None
        Returns:
            None (but plots Figures associated to the static gradient analysis if plotter="paperplotter")
        """
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        
        if not self.load_dir:
            if self.device != 'cpu':    
                device = torch.device('cuda:'+ str(self.device) if torch.cuda.is_available() else 'cpu')  
            else:
                device = torch.device('cpu')    
            self.dataloader, _ = getDataLoaders(
                self.cfg.data.name,
                **self.cfg.data.config,
                is_mse = self.cfg.trainer.criterion == "mse"
            )
            
            alg_cfg =  self.cfg.algorithm.config
            
            input_size, output_size = self.cfg.data.config.input_size, self.cfg.data.config.output_size
            
            self.model = getModel(
                self.cfg.model.name,
                self.cfg.model,
                device,
                input_size = input_size,
                num_classes = output_size,
            ).to(device)
                   
            self.criterion  = getCriterion(self.cfg.trainer.criterion).to(device)
            print(self.cfg.model.config.asynchronous)
            print(self.cfg.data)
            print(self.cfg.algorithm)
            print(self.cfg.model)
            algorithms = {k: getAlgorithm(k, alg_cfg) for k in ('bptt', 'ep')}

            x, y = next(iter(self.dataloader))
            print(device)
            
            # Perform inference
            neurons = self.model.initialize_neurons(x.to(device))
            
            out, ref_neurons = self.model(
                x.to(device), 
                neurons,
                cache_x = True
            )

            # Compute gradients
            gradients = {name: self.compute_gradients(alg, out, x.to(device), y, ref_neurons, device) for name, alg in algorithms.items()}
        else:
            gradients = torch.load(self.load_dir)
            
        # Plot detailed gradients for randomly selected weights
        from .utils import PlotterMeta

        # Paper plotting
        Plotter = PlotterMeta.REGISTRY[self.plotter + 'plotter']
        plotter = Plotter()
        plotter.plot(gradients)
        plt.show()
        
    def compute_gradients(
        self,
        gradient_estimator: Algorithm,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        ref_neurons: List[List[Tensor]],
        device: torch.device
    ) -> Dict[int, Dict[str, Dict[str, Tensor]]]:
        """
        Args:
            gradient_estimator (Algorithm):
                gradient computation algorithm (either EP or BPTT)
            out (Tensor):
                model logits computed on x
            x (Tensor):
                input data
            y (Tensor):
                associated labels
            ref_neurons (List[List[Tensor]]):
                free states of the neurons of all the layers of all blocks
            device (torch.device):
                device at use (cpu or gpu)
            
        Returns:
            gradients (Dict[int, Dict[str, Dict[str, Tensor]]]):
                gradients computed by the algorithm under consideration indexed by block and name
            detailed_gradients (Dict[int, Dict[str, Dict[str, Tensor]]]):
                detailed couterpart of gradients
        """
        
        self.model.zero_grad()
        neurons = copy(ref_neurons)
        gradient_estimator.compute_gradients(out, x, y.to(device), neurons, self.model, self.criterion, cache_x = True)
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters()}

        self.model.zero_grad()
        neurons = copy(ref_neurons)
        detailed_gradients = gradient_estimator.compute_detailed_gradients(out, x, y.to(device), neurons, self.model, self.criterion)
        
        return gradients, detailed_gradients