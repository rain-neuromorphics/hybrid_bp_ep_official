from . import random_symmetric, gaussian_symmetric_random_matrix, custom_kaiming_uniform_
from .base import HopfieldBlockNeuronInitializer, HopfieldBlockWeightInitializer
from torch.nn.init import orthogonal_, kaiming_uniform_
from typing import List
from .vgg import ConvPool
from .resnet import BasicBlock
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from functools import singledispatchmethod

class Zeros(HopfieldBlockNeuronInitializer):
    """
    Neuron initializer class setting all neuron values to zero
    
    Methods
    -------
    __call__:
        initializes neurons, with multiple behavior dispatched
        over the type of the first argument (ConvPool or BasicBlock)
    """
    
    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("Zeros neuron initializer not implemented for block of type {}".format(model))

    @__call__.register(ConvPool)
    def _(self, model: ConvPool, batch_size: int) -> List[Tensor]:
        """
        __call__ method when type(model) == ConvPool
        
        Args:
            model (ConvPool):
                block whose neurons are initialized
            batch_size (int):
                data batch size
        Returns:
            neurons (List[Tensor]):
                initialized neurons
        """
        
        #NOTE: requires specific behavior at it has to handle with pooling
        device = model.device
        neurons = []
        size = model.in_size

        for idx in range(len(model.channels) - 1):
            size = int( (size + 2 * model.paddings[idx] - model.kernels[idx])/ model.strides[idx] + 1 )   # size after conv
            if model.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - model.pools[idx].kernel_size) / model.pools[idx].stride + 1 )  # size after Pool
            neurons += [torch.zeros((batch_size, model.channels[idx + 1], size, size), requires_grad=True, device=device)]

        size = size * size * model.channels[-1]

        for idx in range(len(model.fc)):
            neurons += [torch.zeros((batch_size, model.fc[idx]), requires_grad=True, device=device)]
        
        return neurons

    @__call__.register(BasicBlock)
    def _(self, model: BasicBlock, batch_size: int, size: int, channel: int) -> List[Tensor]:
        """
        __call__ method when type(model) == BasicBlock
        
        Args:
            model (BasicBlock):
                block whose neurons are initialized
            batch_size (int):
                data batch size
                
        Returns:
            neurons (List[Tensor]):
                initialized neurons
        """
        
        return [torch.zeros((batch_size, channel, size, size), device=model.device), torch.zeros((batch_size, channel, size, size), device=model.device)]

class GOE(HopfieldBlockNeuronInitializer):
    """
    Gaussian Orthogonal Ensembles (GOEs) for neurons
    
    Attributes
    ----------
        V (float):
            GOE hyperparameter
    
    Methods
    -------
        __call__:
            initializes neurons, with multiple behavior dispatched
            over the type of the first argument (ConvPool or BasicBlock)
    """
    
    def __init__(self, V: float = 4.9e-5) -> None:
        self.V = V
        
    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("GOE neuron initializer not implemented for block of type {}".format(model))
    
    @__call__.register(ConvPool)
    def _(self, model: ConvPool, batch_size: int) -> List[Tensor]:
        #NOTE: requires specific behavior at it has to handle with pooling
        V = self.V
        if V is None: V=  0.1  # 
        device = model.device
        #symetric orthogonal gaussian
        neurons = []
        size = model.in_size
        N = size * size  # assuming N = size * size for convolution layers

        for idx in range(len(model.channels) - 1):
            size = int( (size + 2 * model.paddings[idx] - model.kernels[idx]) / model.strides[idx] + 1 )   # size after conv
            N = size * size  # update N for the new size

            if model.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - model.pools[idx].kernel_size) / model.pools[idx].stride + 1 )  # size after Pool
                N = size * size  # update N for the new size

            tensor = random_symmetric(
                (model.channels[idx + 1], size, size), torch.sqrt(torch.tensor(V / N)),
                torch.sqrt(torch.tensor(2 * V / N))
            ).repeat((batch_size, 1, 1, 1)).to(device)

            tensor.requires_grad = True
            neurons += [tensor]

        size = size * size * model.channels[-1]

        for idx in range(len(model.fc)):
            neurons += [torch.zeros((batch_size, model.fc[idx]), requires_grad=True, device=device)]

        return neurons

class Weighted_Kaiming(HopfieldBlockWeightInitializer):
    """
    Weighted Kaiming weight initializer
    
    Attributes
    ----------
        alphas (List[float]):
            layer-wise scaling constants used to re-scale standard
            Kaiming weight initialization
    
    Methods
    -------
        __call__:
            initializes weights, with multiple behavior dispatched
            over the type of the first argument (ConvPool or BasicBlock)
    """
    
    def __init__(self, alphas: List[float] = [0.99, 0.99]):
        self.alphas = alphas

    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("Weighted Kaiming weight initializer not implemented for block of type {}".format(model))
    
    @__call__.register(ConvPool)
    def _(self, model: ConvPool) -> None:
        """
        __call__ method when type(model) == ConvPool
        
        Args:
            model (ConvPool):
                block whose weights are initialized
        """
        
        #NOTE: requires specific behavior at it uses layer-wise alpha values
        for i, syn in enumerate(model.synapses):
            alpha = self.alphas[i]
            custom_kaiming_uniform_(syn.weight, alpha)
            if syn.bias is not None:
                if isinstance(syn, nn.Conv2d):
                    gain = 0.5 / np.sqrt(syn.weight.size(1) * syn.kernel_size[0] * syn.kernel_size[1])
                    torch.nn.init.uniform_(syn.bias, -gain, gain)
                if isinstance(syn, nn.Linear):
                    gain = 0.5 / np.sqrt(syn.in_features)
                    torch.nn.init.uniform_(syn.bias, -gain, gain)

class Standard_Kaiming(HopfieldBlockWeightInitializer):
    """
    Standard weight Kaiming initializer
    
    Methods
    -------
    __call__:
        initializes the weights of the block under consideration
    """
    
    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("Standard Kaiming weight initializer not implemented for block of type {}".format(model))
    
    @__call__.register(ConvPool)
    def _(self, model: ConvPool) -> None:
        """
        __call__ method when type (model) == ConvPool
        
        Args:
            model (ConvPool):
                block whose weights are initialized
        """
        
        #NOTE: requires specific behavior at it loops over a "synapses" attribute
        for syn in model.synapses:
            kaiming_uniform_(syn.weight)
            if syn.bias is not None:
                syn.bias.data.fill_(0)
    
    @__call__.register(BasicBlock)
    def _(self, model: BasicBlock) -> None:
        """
        __call__ method when type (model) == BasicBlock
        
        Args:
            model (BasicBlock):
                block whose weights are initialized
        """
        
        def conv_init_(mod: nn.Module) -> None:
            """
            NOTE: Follows standard prescription for ResNets
            (see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
            """
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
        
        def bn_init_(mod: nn.Module) -> None:
            """
            NOTE: Follows standard prescription for ResNets
            (see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
            """
            if isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
            
        model.apply(conv_init_)
        model.apply(bn_init_)
        
        
class Orthogonal_Rand(HopfieldBlockWeightInitializer):
    """
    Orthogonal random weight initializer
    
    Methods
    -------
        __call__:
            initializes the weights
    """
    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("Orthogonal_Rand weight initializer not implemented for block of type {}".format(model))
    
    @__call__.register(ConvPool)
    def _(self, model: ConvPool) -> None:
        """
        Initializes weights when type(model) == ConvPool
        
        Args:
            model (ConvPool):
                block whose weights are initialized
        """
        #NOTE: requires specific behavior at it loops over a "synapses" attribute
        for m in model.synapses:
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

class GOE(HopfieldBlockWeightInitializer):
    """
    Weight initializer based on Gaussian Orthogonal Ensembles (GOE)
    
    Methods
    -------
    __call__:
        initializes the weights of the block under consideration
    """
    
    def __init__(self, V: float = 0.0001, **kwargs):
        self.V = V
    
    @singledispatchmethod
    def __call__(self, model: nn.Module, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError("GOE weight initializer not implemented for block of type {}".format(model))  
    
    @__call__.register(ConvPool)
    def _(self, model: ConvPool) -> None:
        """
        __call__ method when type(model) == ConvPool
        
        Args:
            model (ConvPool):
                block whose weights are initialized
        """
        #NOTE: requires specific behavior at it loops over a "synapses" attribute
        V = self.V
        for m in model.synapses:
            if isinstance(m, torch.nn.Conv2d):
                with torch.no_grad():
                    # handle convolutional layers
                    out_channels, in_channels, kernel_height, _ = m.weight.shape
                    for out_channel in range(out_channels):
                        for in_channel in range(in_channels):
                            m.weight[out_channel, in_channel].copy_(
                                gaussian_symmetric_random_matrix(kernel_height, V)
                            )
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    @__call__.register(BasicBlock)
    def _(self, model: BasicBlock) -> None:
        """
        __call__ method when type(model) == BasicBlock
        
        Args:
            model (BasicBlock):
                block whose weights are initialized
        """
        @torch.no_grad()
        def conv_init_(mod: nn.Module, V: float) -> None:
            """
            NOTE: Normally, Conv2d layers in Resnets are initialized as:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            (see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
            """
            if isinstance(mod, nn.Conv2d):
                # handle convolutional layers
                out_channels, in_channels, kernel_height, _ = mod.weight.shape
                for out_channel in range(out_channels):
                    for in_channel in range(in_channels):
                        mod.weight[out_channel, in_channel].copy_(
                            gaussian_symmetric_random_matrix(kernel_height, V)
                        )
                if mod.bias is not None:
                    mod.bias.data.fill_(0)
        
        def bn_init_(mod: nn.Module) -> None:
            """
            NOTE: Follows standard prescription for ResNets
            (see: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
            """
            if isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
            
        model.apply(lambda mod: conv_init_(mod, self.V))
        model.apply(bn_init_)