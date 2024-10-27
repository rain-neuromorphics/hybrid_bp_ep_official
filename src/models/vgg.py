import torch
from . import IdentityModule, getWeightInit, getNeuronInit
from typing import List, Optional, Callable
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from .base import HopfieldChain, HopfieldBlock, Stepper, StepperMeta
from contextlib import nullcontext
import torch.nn.functional as F
from functools import singledispatchmethod
from src.helpers import NormalizedMSELoss
from types import NoneType
from omegaconf import ListConfig, DictConfig

class VGG(HopfieldChain):
    """
    Models at use in our paper
    
    Attributes (distinct from parent class)
    ----------
    _has_readout (bool):
        says if the model uses an linear layer that reads the last layer of the last block
        to compute logits
    
    blocks (nn.ModuleList):
        list of blocks
    
    readout (nn.Module):
        either identity module (self._has_readout=False) or linear layer (self._has_readout=True)
        
    Methods
    -------
    _make_blocks:
        builds Hopfield blocks
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        num_classes: int = 10,
        **kwargs
    ) -> None:
        """
        Builds a VGG object
        Args:
            cfg (DictConfig):
                hydra config
            device (torch.device):
                device at use
            num_classes (int):
                number of classes for the task at hand
        """
        
        super().__init__(cfg, num_classes)
        
        config = cfg.config
        block_configs = cfg.blocks_config
        self._has_readout = config.readout
        
        self._make_blocks(config, block_configs, device, **kwargs)
        
    def _make_blocks(
        self,
        config: DictConfig,
        block_configs: ListConfig,
        device: torch.device,
        **kwargs
    ) -> None:
        """
        Builds blocks
        
        Args:
            config (DictConfig):
                global configuration of the model (i.e. non block specific)
            block_configs (ListConfig):
                block-wise configurations
            device (torch.device):
                device at use
        """
        
        blocks = []
        if not self._has_readout:
            block_configs[-1]['fc'][-1] = self._num_classes
        else:
            block_configs[-1]['fc'].pop(-1)
            
        for block_config in block_configs:
            blocks.append(
                ConvPool(device, **config, **block_config, **kwargs)
            )
        
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Sequential(nn.Flatten(1), nn.LazyLinear(self._num_classes)) if self._has_readout else IdentityModule()

class ConvPool(HopfieldBlock):
    """
    Blocks at use inside VGG models (ConvPool block = ff block + eb block)
    
    Attributes
    ---------
    _weight_init (HopfieldBlockWeightInitializer):
        weight initializer
    _neuron_init (HopfieldBlockNeuronInitializer):
        neuron initializer
    _stepper (Stepper):
        stepper to execute the fixed point dynamics of the neurons
        
    Methods
    -------
    _make_layers:
        builds layers of a ConvPool block
    _build_conv:
        builds a convolutional layer
    _build_pool:
        builds pooling layer
    _build_fc:
        builds fully connected layer
    _build_feedforward_connection:
        builds feedforward block that precedes
    Phi:
        computes primitive function of a convpool block
        
    Properties
    ----------
    stepper:
        wrapper around _stepper method
    """
    
    def __init__(
        self,
        *args,
        normalization: DictConfig = DictConfig(dict()),
        alphas: List[float] = [0.99, 0.99],
        initialize_weights: bool = True,
        use_autograd: bool = False,
        **kwargs
    ) -> None:    
        """
        Builds a ConvPool object
        
        Args:
            normalization (DictConfig):
                hydra configuration to initialize neuron initializer
            alphas (List[float]):
                default parameters for weighted kaiming initialization
            initialize_weights (bool):
                says whether weights should be initialized when the ConvPool
                object is built
            use_autograd (bool) [IMPORTANT]:
                if True, then default (autograd-based) fixed point stepper is used.
                Otherwise, an *analytical* fixed point stepper is used.
                use_autograd = False ensures speed over generality of the stepper.
        """
        
        super().__init__(*args, **kwargs)
        
        if normalization.weights.name == "weighted_kaiming":
            normalization.weights.config.alphas = alphas
            
        WeightInit, NeuronInit = getWeightInit(normalization.weights.name), getNeuronInit(normalization.neurons.name)
        self._weight_init = WeightInit(**normalization.weights.config)
        self._neuron_init = NeuronInit() if not normalization.neurons.config else NeuronInit(**normalization.neurons.config)
        
        if initialize_weights:
            self._weight_init(self)
                    
        Stepper = StepperMeta.REGISTRY['fixedpointstepper' if use_autograd else 'analyticalfixedpointstepper']
        self._stepper = Stepper(self.initialize_neurons(1), **kwargs)
        
    @property
    def stepper(self) -> Callable:
        """
        Wrapper around _stepper method
        """   
        return self._stepper
         
    def _make_layers(self,
        size: int = 32,
        channels: List[int] = [3, 64, 64],
        kernels: List[int] = [3, 3],
        strides: List[int] = [1, 1],
        paddings: List[int] = [1, 1],
        fc: List[int] = [],
        norm: str = 'b',
        pools: str = '--',
        norm_pool: bool = False,
        **kwargs          
    ) -> None:
        """
        Builds layers of a given ConvPool block
        
        Args:
            size (int):
                input width / height of the block input
            channels (List[int]):
                number of channels / feature maps of each layer (block input included)
            kernels (List[int]):
                list of kernel sizes (one for each convolution operation)
            strides (List[int]):
                list of strides (one for each convolution operation)
            paddings (List[int]):
                list of paddings (one for each convolution operation)
            fc (List[int]):
                list of layer sizes to build fc layers after convolutions
            norm (str):
                normalization layer ("b" = batchnorm, "-" = None)
            pools (str):
                pooling layers coming after each convolution
                ("-" = None, "m" = maxpooling)
            normpool (bool):
                if True, pooling comes after normalization
        """
        
        # necessary to use neuron and weight initializers
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        
        self.synapses = torch.nn.ModuleList()
        self.pools = []
        in_size = size

        for idx, (in_c, out_c, k, s, pad, pool) in enumerate(zip(channels, channels[1:], kernels, strides, paddings, pools)):
            size = self._build_conv(size, in_c, out_c, k, s, pad, bias= not(norm != '-' and idx == 0))
            size = self._build_pool(size, pool)

        self._build_fc(size, fc, channels[-1])
        self._build_feedforward_connection(in_size, norm, norm_pool)
        self.in_size = in_size

    def _build_conv(
        self,
        in_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias=True
    )-> int:
        """
        Builds a single convolutional layer (standard)
        """
        
        self.synapses.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        )
        out_size = int( (in_size + 2 * padding - kernel_size) / stride + 1 )
        return out_size

    def _build_pool(self, in_size: int, pool: str) -> int:
        """
        Builds a single pooling layer (standard)
        """
        
        if pool in ("m", "s"):
            t = 2 if pool == "m" else 1
            self.pools.append(torch.nn.MaxPool2d(t, stride = t))
            out_size = int( (in_size - t) / t + 1 )   # size after Pool
        elif pool == "-":
            self.pools.append(IdentityModule())
            out_size = in_size
        else: raise ValueError("expected `m', `s' or `-' for pooling operation but got {}".format(pool))

        return out_size

    def _build_fc(self, size: int, fc: List[int], channel: int) -> None:
        """
        Builds a fully connected layer (standard)
        """
        
        if len(fc) > 0:
            size = size * size * channel
            fc_layers = [size] + fc
            self._out_size = fc[-1]
            for in_dim, out_dim in zip(fc_layers, fc_layers[1:]):
                self.synapses.append(
                    torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
                )
        else:
            self._out_size = size

    def _build_feedforward_connection(self, size: int, norm: str, norm_pool: bool) -> None:
        """
        Builds the feedforward block which precedes the EB block
        
        Args:
            norm (str):
                the type of normalization used (either batchnorm, layernorm or None)
            normpool (bool):
                if True, pooling comes after normalization
        """
        
        if norm == 'l':
            normalization = torch.nn.LayerNorm(size, device=self.device)
        elif norm =='b':
            normalization = torch.nn.BatchNorm2d(self.synapses[0].out_channels, device=self.device)
        elif norm =='-':
            normalization = IdentityModule()
        else:
            raise ValueError("expected `l', `b' or `-' for normalization operation but got {}".format(self.norm_name))
        
        if norm_pool:
            self.feedforward = nn.Sequential(self.synapses[0], normalization, self.pools[0])
        else:    
            self.feedforward = nn.Sequential(self.synapses[0], self.pools[0], normalization)

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
        """
        Computes the primitive function of a ConvPool block
        
        Args:
            x (Tensor):
                block inputs
            neurons (List[Tensor]):
                all layers of the block
            y (Tensor):
                labels associated to x
            beta (float):
                nudging strength at use
            error_current (Optional[Tensor]):
                error signal used to nudged the block (for all blocks until penultimate)
            loss_fn (Callable):
                loss function used to nudge the last block
            readout (Module):
                model readout if any
        Returns;
            phi (Tensor):
                primitive function computed over the block input
        """
        
        conv_len = len(self.kernels)
        layers = [x.to(self.device)] + neurons
        phi = 0

        base = self.feedforward(layers[0]) * layers[1]
        phi += ((torch.sum(base, dim = (1, 2, 3))).squeeze())

        for idx in range(1, conv_len):
            base = self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx + 1]
            phi += ((torch.sum(base, dim = (1, 2, 3))).squeeze())

        if len(self.fc) > 0:
            for idx in range(conv_len, len(self.synapses)):
                base = self.synapses[idx](layers[idx].view(x.size(0), -1)) * layers[idx + 1]
                phi += torch.sum(base, dim = 1).squeeze()

        if beta != 0.0:
            if error_current is None:
                assert layers[-1].is_leaf, "gradients should not backprop before logits!"
                loss = loss_fn(readout(layers[-1]), y) if type(loss_fn) == CrossEntropyLoss else loss_fn(layers[-1], y.to(self.device).float()).sum(dim=1)
                phi -= beta * loss
            else:
                phi -= beta * (torch.sum(error_current * layers[-1], dim = (1, 2, 3)))

        return phi
    

class AnalyticalFixedPointStepper(Stepper):
    """
    Fixed point stepper which uses analytical neuron updates to go faster
    
    Methods
    -------
    _step:
        computes a single block update
    _compute_update:
        computes the update of a given layer
    grad_post:
        computes the bottom-up contribution of a given layer
    grad_pre:
        computes the top-down contribution of a given layer
    error_signal:
        computes the error signal received by the block
    
    """
    def _step(
        self,
        block: nn.Module,
        x: Tensor,
        neurons: List[Tensor],
        indices = None,
        use_autograd = False,
        **grad_kwargs
    ) -> List[Tensor]:
        """
        Computes a single block update
        Args:
            block (nn.Module):
                block under consideration
            x (Tensor):
                block input
            neurons (List[Tensor]):
                block layers
            indices (List[int]):
                indices of the layers to update
            use_autograd (bool):
                specfies whether autograd is used (True if BPTT is employed)
        Returns:
            neurons (List[Tensor]):
                updated neurons      
        """
        
        layers = [x.to(block.device)] + neurons
        
        with torch.no_grad() if not use_autograd else nullcontext():
            grads = [self._compute_update(block, layers, idx, **grad_kwargs) for idx in indices]
        
        neurons = self.activate(grads, indices)
        assert all([n.is_leaf for n in neurons]) if not use_autograd else all([not n.is_leaf for n in neurons])
        
        return neurons

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
        """
        Computes the update of a given layer
        Args:
            block (nn.Module):
                block under consideration
            layers (List[Tensor]):
                block layers
            idx (int):
                layer index under consideration within the block
            y (Tensor):
                labels associated to the block input
            beta (float):
                nudging strength
            error_current (Tensor):
                error signal used to nudge the block
            loss_fn (Callable):
                loss function used to nudge the last block
            readout (nn.Module):
                model readout if any
        Return:
            updated neurons (Tensor)
        """
        
        grad_post = self.grad_post(block.synapses[idx], layers[idx], idx, block)
            
        if idx == len(block.synapses) - 1:
            grad_pre = torch.zeros_like(grad_post).to(block.device)
            if beta != 0: grad_pre = self.error_signal(loss_fn, readout, error_current, beta, y, layers[-1])
        else:
            grad_pre = self.grad_pre(block.synapses[idx + 1], layers[idx + 2], layers[idx + 1], idx + 1, block)
        
        grad_pre = grad_pre.reshape(grad_post.shape)

        return sum([grad_post, grad_pre])
    
    @singledispatchmethod
    def grad_post(self, mod: nn.Module, *args, **kwargs):
        """
        Bottom-up contribution to the neuron
        """
        raise NotImplementedError("grad_post not implemented for module of type {}".format(mod))
    
    @grad_post.register(nn.Linear)
    def _ (self, mod: nn.Linear, x: Tensor, *args):
        """
        grad_post when type(mod) == nn.Linear
        """
        
        return mod(x.view(x.size(0), -1))

    @grad_post.register(nn.Conv2d)
    def _ (self, mod: nn.Conv2d, x: Tensor, idx: int, block: nn.Module) -> Tensor:
        """
        grad_post when type(mod) == nn.Conv2d
        """
        
        return block.feedforward(x) if idx == 0 else block.pools[idx](mod(x))

    @singledispatchmethod
    def grad_pre(self, mod: nn.Module, *args, **kwargs):
        raise NotImplementedError("grad_pre not implemented for module of type {}".format(mod))

    @grad_pre.register(nn.Linear)
    def _ (self, mod: nn.Linear, x_post: Tensor, x_pre: Tensor, idx, *args) -> Tensor:
        """
        grad_pre when type(mod) == nn.Linear
        """
        
        return F.linear(x_post, mod.weight.data.t())

    @grad_pre.register(nn.Conv2d)
    def _ (self, mod: nn.Conv2d, x_post: Tensor, x_pre: Tensor, idx, block, *args) -> Tensor:
        """
        grad_pre when type(mod) == nn.Conv2d
        """
        
        if not isinstance(block.pools[idx], IdentityModule):
            _, inds = F.max_pool2d(
                mod(x_pre),
                block.pools[idx].kernel_size,
                stride=block.pools[idx].stride,
                return_indices=True
            )
            layer_post = F.max_unpool2d(
                x_post, inds,
                block.pools[idx].kernel_size,
                stride=block.pools[idx].stride
            )
        else:
            layer_post = x_post

        return F.conv_transpose2d(
            layer_post,
            mod.weight.data,
            padding=block.paddings[idx],
            stride=block.strides[idx]
        )
        
    @singledispatchmethod
    def error_signal(self, loss_fn: Callable, *args, **kwargs):
        raise NotImplementedError("error_signal not implemented for loss of type {}".format(loss_fn))  
    
    @error_signal.register(NoneType)
    def _(
        self,
        loss_fn: NoneType,
        readout: Optional[nn.Module],
        error_current: Tensor,
        beta: float,
        y: Tensor,
        x: Tensor
    ) -> Tensor:
        """
        error_signal when no loss_fn
        """
        
        return error_current * beta

    @error_signal.register(NormalizedMSELoss)
    def _(
        self,
        loss_fn: NormalizedMSELoss,
        readout: Optional[nn.Module],
        error_current: Tensor,
        beta: float,
        y: Tensor,
        x: Tensor
    ) -> Tensor:
        """
        error_signal when type(loss_fn) == NormalizedMSELoss
        """
        
        grad_pre = beta * (y - x) 
        if isinstance(readout, nn.Linear): 
            grad_pre = F.linear(grad_pre, readout[1].weight.data.t())
        return grad_pre

    @error_signal.register(nn.CrossEntropyLoss)
    def _(
        self,
        loss_fn: nn.CrossEntropyLoss,
        readout: Optional[nn.Module],
        error_current: Tensor,
        beta: float,
        y: Tensor,
        x: Tensor
    ) -> Tensor:
        """
        error_signal when type(loss_fn) == NormalizedMSELoss
        """
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