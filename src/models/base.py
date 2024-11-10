from . import clamp
from src.helpers import copy
from typing import List, Optional, Callable, Tuple
from torch import device
import torch.nn as nn
from torch import Tensor
import abc
from . import HopfieldBlockMeta, StepperMeta, NeuronInitMeta, WeightInitMeta, ModelMeta
from omegaconf import DictConfig
import torch.nn.functional as F
import torch
from omegaconf import ListConfig, DictConfig
import numpy as np

class HopfieldChain(torch.nn.Module, metaclass=ModelMeta):
    """
    Implements ff-EBMs (called here "Hopfield chains")
    
    Attributes
    ----------
    config (DictConfig):
        hydra configuration
    _num_iterations (int):
        number of iterations which apply to the neurons by default
    _num_classes (int):
        number of classes with the task of interest
    
    Methods
    -------
    _make_blocks:
        builds each block comprising each a feedforward * and * an energy-based block
        (feedforward transformation is implicitly absorbed into the definition of the energy,
        check out our paper for details)
    forward:
        ff-EBM inference procedure
    initialize_neurons:
        self explanatory
    initialize_weights:
        self explanatory
    
    Properties
    ----------
    has_readout:
        says if the ff-EBM as a readout (i.e. when using the cross entropy loss function)
    num_iterations:
        number of fixed-point iterations which are used by default      
    """
    
    def __init__(self, cfg: DictConfig, num_classes: int = 10) -> None:
        super(HopfieldChain, self).__init__()
        config = cfg.config
        self._num_iterations = config.T1
        self._num_classes = num_classes

    @abc.abstractmethod
    def _make_blocks(self, config: DictConfig, block_configs: ListConfig, device: torch.device, **kwargs) -> None:

        raise NotImplementedError("_make_blocks method of HopfieldChain subclass is not implemented")
    
    def forward(
        self, input: Tensor, neurons: List[List[Tensor]], num_iterations: Optional[int]= None, **kwargs
    ) -> Tuple[Tensor, List[List[Tensor]]]:
        """
        ff-EBM inference procedure
        Args:
            input (Tensor): input (data) tensor
            neurons (List[List[Tensor]]): block states as a list of layers, where each layer is itself a list of tensors
            num_iterations (Optional[int]): number of iterations used to compute equilibrium
        Returns:
            h (Tensor): logits / last layer
            neurons (List[List[Tensor]]): updated block states
        """
        
        if not num_iterations: num_iterations = self._num_iterations

        h = input
        
        for i, (b, n) in enumerate(zip(self.blocks, neurons)):
            neurons[i] = b(h, n, num_iterations, **kwargs)
            h = neurons[i][-1]
                
        if self._has_readout: h = self.readout(h)
        return h, neurons

    def initialize_neurons(self, input: Tensor) -> List[List[Tensor]]:
        """
        Initializes neurons
        Args:
            input (Tensor): input (data) tensor
        Returns:
            neurons (List[List[Tensor]]): initialized block states
        """
                                    
        batch_size = input.size(0)
        return [b.initialize_neurons(batch_size) for i, b in enumerate(self.blocks)]

    def initialize_weights(self) -> None:
        """
        Initializes weights
        """
        
        for b in self.blocks:
            b.initialize_weights()

    @property
    def has_readout(self) -> bool:
        return self._has_readout

    @property
    def num_iterations(self) -> int:
        return self._num_iterations
    
class HopfieldBlock(nn.Module, metaclass=HopfieldBlockMeta):
    """
    Class implementing a single Hopfield block.
    Using the terminology of our paper, a Hopfield block comprises two blocks: one feedforward block and one EB block.
    The feedforward block is * implicitly * absorbed into the definition of the energy function (check out our paper
    for details)
    
    Attributes
    ----------
    device (torch.device):
        device where the model is located
    tol (Optionat[float]):
        tolerance value optionally used if a tolerance-based criterion
        is used to execute the dynamics of the neurons
    
    Methods
    -------
    forward:
        inference procedure inside a block
    _converged:
        check if neuron dynamics are converged
    compute_trajectory:
        computes the trajectory of the neurons of the block
    Phi:
        computes the primitive function 
    initialize_neurons:
        initialize block neurons
    initialize_weights:
        initialize block weights
    make_layers:
        build layers of the block
    _build_feedforward_connection:
        build the feedforward block which is implicitly absorbed into the definition of the whole block
    
    Properties
    ----------
    block_x:
        block input
    stepper:
        stepper at use inside the block
    out_size:
        size of the block output
    """
    
    def __init__(
        self,
        device: device,
        tol: Optional[float] = None,
        **kwargs
    ) -> None:    

        super(HopfieldBlock, self).__init__()
        
        self.device = device
        self.tol = tol
        self._make_layers(**kwargs)
    
    def forward(
        self,
        x: Tensor,
        neurons: List[Tensor],
        num_iterations: int,
        cache_x: bool = False,
        tol: bool = True,
        **kwargs
    ) -> List[Tensor]:
        """
        Inference procedure inside a block
        Args:
            x (Tensor):
                block input (therefore, input to the feedforward transformation that is paired with the EB block) 
            neurons (List[Tensor]):
                block neurons, where each element of the list is a layer 
            num_iterations (int):
                number of iterations used to compute equilibrium inside the block
            cache_x (bool):
                specifies whether we want to explicitly cache the block input (when applying EP especially)
            tol (bool):
                specifies if a tolerance-based criterion is used to stop the fixed point iteration procedure
        Returns:
            neurons (List[Tensor]): updated block neurons
        """
        
        tol = tol and self.tol is not None
        for i in range(num_iterations):
            neurons_tmp = self.stepper(self, x, neurons[:], **kwargs)
            if tol and self._converged(neurons_tmp, neurons):
                neurons = neurons_tmp
                break
            else:
                neurons = neurons_tmp
            
        if cache_x: self._block_x = x.detach().clone().requires_grad_()
        return neurons

#    def _converged(self, neurons_tmp: List[Tensor], neurons: List[Tensor], epsilon: float = 1e-22) -> bool:
#        measure = np.mean([(abs(n_tmp - n) / abs(n).clip(min=epsilon)).mean().item() for n_tmp, n in zip(neurons_tmp, neurons)])
#        return measure < self.tol

    @torch.no_grad()
    def _converged(self, neurons_tmp: List[Tensor], neurons: List[Tensor], epsilon: float = 1e-6) -> bool:
        """
        Checks if neuron dynamics are converged per some criterion
        Args:
            neurons_tmp (List[Tensor]):
                neurons at current time step
            neurons (List[Tensor]):
                neurons at previous time step
            epsilon (float):
                value added at the denominator to prevent division by 0
        Returns:
            converged (bool):
                says if the dynamics are converged
        """
        
        measure = np.mean([(torch.norm(n_tmp - n) / torch.norm(n).clip(min=1e-6)).item() for n_tmp, n in zip(neurons_tmp, neurons)])
        return measure < self.tol 
    
    def compute_trajectory(
        self,
        x: Tensor,
        neurons: List[Tensor],
        num_iterations: int,
        retain_grad: bool = False,
        **kwargs
    ) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """
        Compute the neurons trajectory throughout equilibrium computation
        Args:
            x (Tensor):
                block input (therefore, input to the feedforward transformation that is paired with the EB block)
            neurons (List[Tensor]):
                block neurons, where each element of the list is a layer
            num_iterations (int):
                number of iterations used to compute equilibrium inside the block
            retain_grad (bool):
                specifies whether we want to track neuron gradients (necessary to track detailed AD gradients)
        
        Returns:
            neurons (List[Tensor]):
                the updated block neurons
            trajectory (List[List[Tensor]]):
                the whole trajectory of all the neurons of the block       
        """
        
        trajectory = []
        if retain_grad:
            for n in neurons: n.requires_grad = True

        trajectory += [copy(neurons)]
        
        for _ in range(num_iterations):
            neurons = self.stepper(self, x, neurons, **kwargs)
            trajectory += [copy(neurons)]
        return neurons, trajectory

    @abc.abstractmethod
    def Phi(self, 
        x: Tensor,
        neurons: List[Tensor],
        y: Optional[Tensor] = None,
        beta: float=0.0, 
        error_current: Optional[Tensor] = None,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        readout: Optional[nn.Module] = None,
    ) -> Tensor:
        """
        Computes the * primitive function * (i.e. some reparametrization of the energy function, see our paper)
        Args:
            x (Tensor):
                block input
            neurons (List[Tensor]):
                block neurons
            y (Optional[Tensor]):
                ground-truth labels associated to current data batch
            beta (float):
                nudging strength
            error_current (Optional[Tensor]):
                error current which is used to nudge the block
            loss_fn (Optional[Callable[[Tensor, Tensor], Tensor]]):
                loss function at use
            readout (Optional[nn.Module]):
                linear readout to probe the last layer of the last block
        Returns:
            Phi (tensor):
                the primitive function value over the current batch
        """
        
        raise NotImplementedError("Phi method of HopfieldBlock is not implemented")   

    def initialize_neurons(self, batch_size: int) -> List[Tensor]:
        """
        Initializes block neurons
        Args:
            batch_size (int):
                size of the data batch
        Returns:
            neurons (List[Tensor]):
                initialized block neurons
        """
        
        neurons = self._neuron_init(self, batch_size)
        self._neurons_trajectory = []
        return neurons

    def initialize_weights(self) -> None:
        """
        Initializes block weights
        """
        
        self._weight_init(self)

    @abc.abstractmethod
    def _make_layers(self):
        raise NotImplementedError("_make_layers method of HopfieldBlock is not implemented")

    @abc.abstractmethod
    def _build_feedforward_connection(self, size: int) -> None:
        raise NotImplementedError("_build_feedforward_connection method of HopfieldBlock is not implemented")   

    @property
    def block_x(self) -> Tensor:
        return self._block_x

    @property
    def stepper(self) -> Callable:
        raise NotImplementedError("_stepper method of HopfieldBlock is not implemented") 

    @property
    def out_size(self) -> int:
        return self._out_size

class Stepper(metaclass=StepperMeta):
    """
    Class to implement root finding algorithms to compute equilibrium states
    
    Attributes
    ----------
    activation (Callable):
        activation function
    asynchronous (bool):
        specifies if asynchronous updates should be used
    clamp_last(bool):
        specifies whether the activation function should be applied to the last layer
    activations (List[int]):
        by definition, activations[i] = 1 if activation function is applied to layer[i]
        (0 otherwise)
    indices (List[int]):
        [0, ..., len(neurons) - 1]
    
    Methods
    -------
    __call__:
        executes a single synchronous or asynchronous steps of * all * the neurons,
        wrapping around the _step method
    _step:
        executes a single step on specified neurons
    activate:
        applies the activation function on layers where it should apply 
    """
    
    def __init__(
        self,
        neurons: List[Tensor],
        activation: int = 0.5,
        asynchronous: bool = True,
        clamp_last: bool = False,
        **kwargs
    ) -> None:
        
        self.activation = clamp(activation=activation)
        self.asynchronous = asynchronous
        self.clamp_last = clamp_last
        self.activations = [1 for _ in range(len(neurons))]
        if not self.clamp_last: self.activations[-1] = 0
        self.indices = list(range(len(neurons)))
        
    def __call__(self,
        block: nn.Module,
        x: Tensor,
        neurons: List[Tensor],
        asynchronous: bool = False,
        backwards: bool = False,
        **grad_kwargs
    ) -> List[Tensor]:
        """
        Executes a single synchronous or asynchronous steps of * all * the neurons,
        wrapping around the _step method
        
        Args:
            block (nn.Module):
                block under consideration
            x (Tensor):
                block input
            neurons (List[Tensor]):
                block layers
            asynchronous (bool):
                specifies if dynamics should be executed asynchronously
            backwards (bool):
                specifies for asynchronous dynamics whether to start with
                odd (backwards=True) or even (backwards=False) layers
        
        Returns:
            neurons (List[Tensor]):
                updated block layers
        """
        
        asynchronous = asynchronous or self.asynchronous
        if asynchronous:
            i, j = (1, 0) if backwards else (0, 1)
            neurons[i::2] = self._step(block, x, neurons, indices=self.indices[i::2], **grad_kwargs)    
            neurons[j::2] = self._step(block, x, neurons, indices=self.indices[j::2], **grad_kwargs) 
        else:
            neurons = self._step(block, x, neurons, indices=self.indices, **grad_kwargs)
                
        return neurons

    def _step(self, *args, **kwargs):
        """Abstract _step method"""
        
        raise NotImplementedError("_step method not implemented in Stepper subclass")

    def activate(
        self,
        grads: List[Tensor],
        indices: Optional[List[int]] =  None
    ) -> List[Tensor]:
        """
        Applies activation function layerwise where specified
        
        Args:
            grads (List[Tensor]):
                preactivations
            indices (Optional[List[int]]):
                indicates where the activation function should be applied
        Returns:
            updated list of activations (List[Tensor])    
        """
        
        if indices == None:
            return [self.activation(g) for g in grads]
        else:
            activations = [self.activations[i] for i in indices]
            return [self.activation(g) if activations[i] == 1 else g for i, g in enumerate(grads)]

class FixedPointStepper(Stepper):
    """
    Fixed point stepper where primitive function gradients are computed by
    automatic differentiation (instead of analytical formulas)
    """
    
    def _step(
        self,
        block: nn.Module,
        x: Tensor,
        neurons: List[Tensor],
        use_autograd: bool = False,
        indices: Optional[List[int]] = None,
        **grad_kwargs
    ) -> List[Tensor]:
        """
        Default fixed point stepper core function
        
        Args:
            block (nn.Module):
                block under consideration
            x (Tensor):
                block input
            neurons (List[Tensor]):
                block neurons
            use_autograd (bool):
                says whether the primitive function differentiation
                must generate a graph to be differentiated through
                (to enable BPTT)
            indices (List[int]):
                indices of the block layers to be updated
                
        Returns:
            neurons (List[Tensor]):
                updated layers
        """
        
        active_neurons = neurons if indices == None else [neurons[i] for i in indices]

        init_grads = torch.tensor(
            [1 for _ in range(x.size(0))],
            dtype=torch.float,
            device=block.device,
            requires_grad=True
        )

        grads = torch.autograd.grad(
            block.Phi(x, neurons, **grad_kwargs),
            active_neurons,
            create_graph = use_autograd,
            grad_outputs = init_grads
        )
        
        neurons = self.activate(grads, indices)

        if not use_autograd:                            # when using EP with autograd to compute neural dynamics
            assert all([n.is_leaf for n in neurons])
            for n in neurons: n.requires_grad = True
        else:                                           # when using BPTT 
            assert all([not n.is_leaf for n in neurons])

        return neurons

class HopfieldBlockNeuronInitializer(metaclass=NeuronInitMeta):
    """
    Abstract class to initialize neurons inside Hopfield blocks
    """
    
    def __init__(self, **kwargs) -> None:
        ...
        
    @abc.abstractmethod
    def __call__(self, model: nn.Module, batch_size: int) -> List[Tensor]:
        raise NotImplementedError("__call__ method of HopfieldBlockNeuronInitializer is not implemented")

class HopfieldBlockWeightInitializer(metaclass=WeightInitMeta):
    """
    Abstract class to initialize weights inside Hopfield blocks
    """
    
    def __init__(self, **kwargs):
        ...

    @abc.abstractmethod
    def __call__(self, model: nn.Module) -> None:
        raise NotImplementedError("__call__ method of HopfieldBlockWeightInitializer is not implemented")

