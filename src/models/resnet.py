from __future__ import annotations
from .base import HopfieldChain, HopfieldBlock
from .vgg import AnalyticalFixedPointStepper
from typing import List, Optional, Callable
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from omegaconf import DictConfig
from . import StepperMeta, getWeightInit, getNeuronInit
import torch.nn.functional as F
from functools import singledispatchmethod
from src.helpers import NormalizedMSELoss
from types import NoneType

class ResNet(HopfieldChain):
    def __init__(self, cfg: DictConfig, device: torch.device, num_classes: int = 10, input_size: int = 32, **kwargs) -> None:
        super(HopfieldChain, self).__init__()
        config = cfg.config
        self._num_iterations = config.T1
        self._num_classes = num_classes
        self._input_size = input_size
        self._has_readout = True
        self.in_planes = 64
        self._make_blocks(config, device, **kwargs)

    def _make_blocks(self, config: DictConfig, device: torch.device, **kwargs) -> None:
        blocks = []
        
        first_module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)         # not sure if this is a good idea
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        size = self._input_size // 4                                                                                #NOTE: e.g. size = 8
        blocks += self._make_layer(size, device, 64, self._resnet_cfg[0], extra_module=first_module, **config, **kwargs)      # size = 8
        blocks += self._make_layer(size // 2, device, 128, self._resnet_cfg[1], stride=2, **config, **kwargs)                 # size = 4
        blocks += self._make_layer(size // 4, device, 256, self._resnet_cfg[2], stride=2, **config, **kwargs)                 # size = 2
        blocks += self._make_layer(size // 8, device, 512, self._resnet_cfg[3], stride=2, **config, **kwargs)                 # size = 1
        
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1), nn.Linear(512, self._num_classes))
    
    def _make_layer(
        self,
        size: int,
        device: torch.device,
        planes: int,
        num_layers: int,
        stride: int = 1,
        extra_module: Optional[nn.Module] = None,
        **kwargs
    ) -> List[BasicBlock]:

        blocks = []
        blocks += [BasicBlock(device, size=size, channels=[self.in_planes, planes], stride=stride, extra_module=extra_module, **kwargs)]
        self.in_planes = planes
        
        for _ in range(1, num_layers):
            blocks += [BasicBlock(device, size=size, channels=[planes, planes], **kwargs)]
            
        return blocks
    
    @property
    def _resnet_cfg(self) -> List[int]:
        raise NotImplementedError('ResNet configuration not specified')

class ResNet18(ResNet):
    CFG = [2, 2, 2, 2]
    
    @property
    def _resnet_cfg(self) -> List[int]:
        return self.CFG

class ResNet34(ResNet):
    CFG = [3, 4, 6, 3]
    
    @property
    def _resnet_cfg(self) -> List[int]:
        return self.CFG

class BasicBlock(HopfieldBlock):
    def __init__(
        self,
        *args,
        size: int = 32,
        normalization: DictConfig = DictConfig(dict()),
        initialize_weights: bool = True,
        **kwargs
    ) -> None:
        
        self.has_extra_module = False
        super().__init__(*args, **kwargs)
        
        WeightInit = getWeightInit(normalization.weights.name)
        Zeros = getNeuronInit("zeros")
        self._weight_init = WeightInit(**normalization.weights.config)
        self._neuron_init = Zeros()
        
        if initialize_weights:
            self._weight_init(self)
                      
        self.size = size           
        Stepper = StepperMeta.REGISTRY['residualanalyticalfixedpointstepper']
        self._stepper = Stepper(self.initialize_neurons(1), **kwargs)
        
    @property
    def stepper(self) -> Callable:   
        return self._stepper
    
    def initialize_neurons(self, batch_size: int) -> None:
        neurons = self._neuron_init(self, batch_size, self.size, self.channels[1])
        self._neurons_trajectory = []
        return neurons
        
    def _make_layers(self,
        size: int = 32,
        channels: List[int] = [3, 64],                                          # [input channels, output channels]
        stride: int = 1,                                                        # stride applied to first conv
        **kwargs          
    ) -> None:
        
        # Formatting things as requested by ConvPool class and subsequent weight & neurons initializers
        self.channels = [channels[0]] + 2 * [channels[1]]                       # e.g. [3, 64, 64]
        self.kernels = [3, 3]                                                   #NOTE: always the case, except for the very first conv of the ResNet architecture
        self.strides = [stride] + [1]                                           #NOTE: second conv always has stride 1
        self.paddings = [1, 1]                                                  #NOTE: always the case
        self.in_size = size
        
        self.synapses = torch.nn.ModuleList()

        for _, (in_c, out_c, k, s, pad) in enumerate(zip(self.channels, self.channels[1:], self.kernels, self.strides, self.paddings)):
            self._build_conv(in_c, out_c, k, s, pad)          #NOTE: no pooling anymore

        self._build_feedforward_connection(channels[1], **kwargs)
        self._build_downsample(stride, channels)

    def _build_conv(
        self,
        in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    )-> None:
        
        self.synapses.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )

    def _build_downsample(self, stride: int, channels: List[int]) -> None:
        self.downsample = None
        if stride != 1 or channels[0] != channels[1]:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels[1]),
            )
        
    def Phi(self, 
        x: Tensor,
        neurons: List[Tensor],
        y: Optional[Tensor] = None,
        beta: float = 0.0, 
        error_current: Optional[Tensor] = None,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        readout: Optional[nn.Module] = None,
        **kwargs
    ) -> Tensor:

        layers = [x.to(self.device)] + neurons

        if len(neurons) > 3:
            raise ValueError("There are {} layers inside ResNet block instead of three (counting input layer)".format(len(neurons)))
        
        # computing standard energy terms
        base = self.feedforward(layers[0]) * layers[1] + self.synapses[1](layers[1]) * layers[2]  
        
        # adding skip connection
        identity = self.feedforward[0](layers[0]) if self.has_extra_module else layers[0]
        if self.downsample is not None:
            identity = self.downsample(identity)
        base += identity * layers[2]                                                              
        
        phi = ((torch.sum(base, dim = (1, 2, 3))).squeeze())

        if beta != 0.0:
            if error_current is None:
                assert layers[-1].is_leaf, "gradients should not backprop before logits!"
                if type(loss_fn) != CrossEntropyLoss:
                    raise TypeError("Loss function at use inside BasicBlock is {} (expected: CrossEntropyLoss)".format(type(loss_fn)))
                loss = loss_fn(readout(layers[-1]), y)
                phi -= beta * loss
            else:
                phi -= beta * (torch.sum(error_current * layers[-1], dim = (1, 2, 3)))

        return phi

    def _build_feedforward_connection(self, channel: int, extra_module: Optional[nn.Module] = None, **kwargs) -> None:
        layers = [self.synapses[0], torch.nn.BatchNorm2d(channel, device=self.device)]
        
        if extra_module is not None:
            layers = [extra_module] + layers
            self.has_extra_module = True
        
        self.feedforward = nn.Sequential(*layers)
        
class ResidualAnalyticalFixedPointStepper(AnalyticalFixedPointStepper):

    def _compute_update(
        self,
        block: nn.Module,
        layers: List[Tensor],
        idx: int,
        y: Optional[Tensor] = None,
        beta: float = 0.0, 
        error_current: Optional[Tensor] = None,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        readout: Optional[nn.Module] = None,
        **kwargs
        ) -> Tensor:
                                                                                #NOTE: adding block input for residual connection
        grad_post = self.grad_post(block.synapses[idx], layers[idx], idx, block, layers[0])                     
            
        if idx == len(block.synapses) - 1:
            grad_pre = torch.zeros_like(grad_post).to(block.device)
            if beta != 0: grad_pre = self.error_signal(loss_fn, readout, error_current, beta, y, layers[-1])
        else:
            grad_pre = self.grad_pre(block.synapses[idx + 1], layers[idx + 2], layers[idx + 1], idx + 1, block)
        
        grad_pre = grad_pre.reshape(grad_post.shape)

        return sum([grad_post, grad_pre])
    
    @singledispatchmethod
    def grad_post(self, mod: nn.Module, *args, **kwargs):
        raise NotImplementedError("grad_post not implemented for module of type {}".format(mod))
    
    @grad_post.register(nn.Linear)
    def _ (self, mod: nn.Linear, x: Tensor, *args):
        return mod(x.view(x.size(0), -1))

    @grad_post.register(nn.Conv2d)
    def _ (self, mod: nn.Conv2d, x: Tensor, idx: int, block: nn.Module, block_input: Tensor):
        if idx == 0:
            return block.feedforward(x)
        else:
            out = mod(x)                        #NOTE: there is no pooling anymore
            if idx == len(block.synapses) - 1:
                identity = block.feedforward[0](block_input) if block.has_extra_module else block_input
                if block.downsample is not None:
                    identity = block.downsample(identity)
                out += identity                 #NOTE: residual connection
            return out
                
    @singledispatchmethod
    def grad_pre(self, mod: nn.Module, *args, **kwargs):
        raise NotImplementedError("grad_pre not implemented for module of type {}".format(mod))

    @grad_pre.register(nn.Linear)
    def _ (self, mod: nn.Linear, x_post: Tensor, x_pre: Tensor, idx, *args):
        return F.linear(x_post, mod.weight.data.t())

    @grad_pre.register(nn.Conv2d)
    def _ (self, mod: nn.Conv2d, x_post: Tensor, x_pre: Tensor, idx, block, *args):
        return F.conv_transpose2d(              #NOTE: there is no pooling anymore
            x_post,         
            mod.weight.data,
            padding=block.paddings[idx],
            stride=block.strides[idx]
        )
        
    @singledispatchmethod
    def error_signal(self, loss_fn: Callable, *args, **kwargs):
        raise NotImplementedError("error_signal not implemented for loss of type {}".format(loss_fn))  
    
    @error_signal.register(NoneType)
    def _(self, loss_fn: NoneType, readout: Optional[nn.Module], error_current: Tensor, beta: float, y: Tensor, x: Tensor):
            return error_current * beta

    @error_signal.register(nn.CrossEntropyLoss)
    def _(self, loss_fn: nn.CrossEntropyLoss, readout: Optional[nn.Module], error_current: Tensor, beta: float, y: Tensor, x: Tensor):
        with torch.enable_grad():
            x.requires_grad = True
            assert x.is_leaf, "Error signal computation must use leaf variables only"
            grad_pre = - beta * torch.autograd.grad(
                    loss_fn(readout(x), y).sum(),
                    x,
                    only_inputs=True,  # Do not backpropagate further than the input tensor!
                    create_graph=False,
                )[0]
        return grad_pre
    
    
