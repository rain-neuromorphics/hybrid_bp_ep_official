import torch
from src.models.base import HopfieldChain
from typing import Dict, Any, Tuple, Callable, Optional, List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
import torch.distributed as dist
from torch import Tensor, device
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from src.algorithms.base import Algorithm
import os
import numpy as np
import abc
import matplotlib.pyplot as plt
from omegaconf import DictConfig

class NormalizedMSELoss(MSELoss):
    """
    Custom MSE loss function with default forward method being overriden 
    """
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 0.5 * F.mse_loss(input, target, reduction=self.reduction)


_CRITERIA = {
    'cross_entropy': torch.nn.CrossEntropyLoss,
    'mse': NormalizedMSELoss
}

_SCHEDULERS = {
    'step': torch.optim.lr_scheduler.StepLR, 
    'multi_step': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'cosine_warmup': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}

_OPTIMIZERS = {
    'sgd': torch.optim.SGD, 
    'adam': torch.optim.Adam,
}

class PlotterMeta(type):
    """
    Metaclass registering the different plotters at use
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(PlotterMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
    
class Plotter(metaclass=PlotterMeta):
    """
    Defines the parent plotter class used for the static gradient analysis
    
    Methods
    -------
    plot:
        produces the desired set of plots (e.g. detailed gradient curves, bar plots)
    plot_curves:
        plots detailed gradient curves for all blocks
    _plot_curves:
         plots detailed gradient curves for a given block and a given algorithm
    _plot_single_curves:
        plots detailed gradient curves for a given parameter or neuron
    """
    
    LINESTYLES: Dict[str, str] = {
        'bptt': '-',
        'ep': '--'
    }
    SMALL_SIZE: int = 8
    MEDIUM_SIZE: int = 15
    BIGGER_SIZE: int = 18    
    
    def plot(self, *args, **kwargs) -> None:
        self.plot_curves(*args, **kwargs)
    
    @abc.abstractmethod
    def plot_curves(self, *args, **kwargs)-> None:
        raise NotImplementedError("plot_curves method not implemented")
        
    @abc.abstractmethod
    def _plot_curves(self, *args, **kwargs) -> None:    
        raise NotImplementedError("_plot_curves method not implemented")

    def _plot_single_curves(self, *args, **kwargs) -> None:
        raise NotImplementedError("_plot_single_curves method not implemented")
  
class DefaultPlotter(Plotter):
    """
    Default plotter used for fast static gradient analysis (not the one used to produce the curves of the paper)
    """
    
    def plot(self, gradients, **kwargs) -> None:
        """
        Same as parent class 
        """
        
        self.plot_curves({k : v[1] for k, v in gradients.items()}, type='params', **kwargs)
        
    def plot_curves(
        self,
        detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]],
        type: str = 'neurons',
        fontsize: int = 10,
        **kwargs
    )-> None:
        """
        Plot detailed gradient curves either for parameters or neurons (i.e. computing d loss / d neuron_i at each step of the unfolded computational graph)
        
        Args:
            detailed_gradients (Dict[int, Dict[str, Dict[str, Tensor]]]):
                detailed gradients for all layers of all blocks, for parameters and neurons
            type (str):
                specifies if we want to plot neuron gradients or parameter gradients
            fontsize (int):
                default font size for the generated plots
        """
        
        plt.rcParams.update({'font.size': fontsize})

        self.figs = []
        self.axes = []
        self.idx_dims = dict()

        for idx_alg, (alg_name, detailed_grads) in enumerate(detailed_gradients.items()):
            for idx_block in range(len(detailed_grads.items())):
                self._plot_curves(idx_alg, alg_name, idx_block, detailed_grads[idx_block][type], **kwargs)

        for fig in self.figs:
            plt.subplots_adjust(hspace = 0.5)
            fig.tight_layout()
        
    def _plot_curves(
        self,
        idx_alg: int,
        alg_name: str,
        idx_block: int,
        grads: Dict[str, Tensor]
    ) -> None:   
        """
        Plots detailed gradients for a given algorithm  and a given block
        
        Args:
            idx_alg (int):
                index of the algorithm under consideration (O or 1)
            alg_name (str):
                name of the algorithm ("ep" or "bptt")
            idx_block (int):
                index of the block
            grads (Dict[str, Tensor]):
                detailed gradients indexed by name
        """
        
        if idx_alg == 0:
            numel = len(grads)
            ncols = int(np.ceil(np.sqrt(numel)))
            nrows = numel // ncols + int(numel % ncols > 0)
            # grads_new = {k: v for k, v in grads.items() if k.split(".")[-1] == 'weight'}
            fig_tmp, axes_tmp = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
            axes_tmp = axes_tmp.reshape(-1)
            for ax in axes_tmp[numel:]: ax.remove()
            self.figs += [fig_tmp]
            self.axes += [axes_tmp]

        for idx_param, (param_name, param_grads) in enumerate(grads.items()):
            self._plot_single_curves(idx_alg, self.axes[idx_block][idx_param], param_grads, alg_name, idx_block, param_name)

    def _plot_single_curves(
        self,
        idx_alg: int,
        ax,
        grads: Tensor,
        alg_name: str,
        idx_block: int,
        param_name: str,
        linewidth: int = 4,
        num_samples: int = 9,
        color: Optional[np.ndarray] = None
    ) -> None:
        """
        Plots detailed gradients for a given layer
        
        Args:
            idx_alg (int):
                same as in _plot_curves
            ax:
                matplotlib axis object under consideration
            grads (Tensor):
                detailed gradients of the layer under consideration
            alg_name (str):
                same as in _plot_curves
            idx_block (int):
                same as in _plot_curves
            param_name (str):
                name of the layer under consideration
            linewidth (int):
                width of the curves
            num_samples (int):
                number of neurons / parameters being randomly sampled for the plots
        """
        
        grads = grads.view(grads.size(0), -1)

        if idx_alg == 0:
            self.idx_dims[(idx_block, param_name)] = [np.random.randint(grads.size(1)) for _ in range(num_samples)]
        
        for idx_sample, idx_dim in enumerate(self.idx_dims[idx_block, param_name]):
            ax.plot(grads[:, idx_dim].detach().cpu().numpy(), 
                    color='C'+ str(idx_sample), 
                    linewidth = linewidth,
                    linestyle = self.LINESTYLES[alg_name],
                    alpha = 0.5
                    )

        ax.set_xlabel('t')
        ax.set_title('block {}-{}'.format(idx_block, param_name))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
class PaperPlotter(Plotter):
    """
    Plotter which was used for our paper
    """
    
    def plot(self, gradients, **kwargs) -> None:
        """
        Plots detailed gradients and bars (Fig. 3 and 5 of the camera-ready version of our paper)
        """
        
        self.build_colormap({k : v[1] for k, v in gradients.items()})
        self.plot_curves({k : v[1] for k, v in gradients.items()})
        self.plot_bars({k : v[1] for k, v in gradients.items()})
    
    def build_colormap(self, detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]]) -> None:
        """
        Builds a colormap gracefully spanning between blue and yellow through red
        """
        
        n_layers = sum([len(list(filter(lambda x: x.split(".")[-1] == 'weight' and x.split(".")[0] == 'synapses', list(g['params'].keys())))) for g in detailed_gradients['ep'].values()])
        cmap = plt.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, n_layers))
        
        # build dictionary mapping param names to colors
        self.colors = {}
        counter = 0
        for idx_block in range(len(detailed_gradients['ep'].items())):
            for param_name in detailed_gradients['ep'][idx_block]['params'].keys():
                if param_name.split(".")[-1] == 'weight' and param_name.split(".")[0] == "synapses":
                    self.colors['block{}.'.format(idx_block) + param_name] = colors[counter]
                    counter += 1
                
    def plot_curves(
        self,
        detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]],
    )-> None:
        """
        Same as parent class
        """
        
        import os
        # '/opt/anaconda3/lib/python3.9/site-packages/latex'
        plt.rc('font', size=self.SMALL_SIZE)           # controls default text sizes
        plt.rc('axes', titlesize=self.BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('xtick', labelsize=self.MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('axes', labelsize=self.BIGGER_SIZE)    # fontsize of the x and y labels
        self.idx_dims = dict()

        for idx_alg, (alg_name, detailed_grads) in enumerate(detailed_gradients.items()):
            self.counter = 0
            if idx_alg == 0:
                n_layers_per_block = [len(list(filter(lambda x: x.split(".")[-1] == 'weight' and x.split(".")[0] == 'synapses', list(g['params'].keys())))) for g in detailed_grads.values()]
                nrows = max(n_layers_per_block)
                ncols = len(detailed_grads)
                self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, nrows), layout="constrained")
                self.fig.tight_layout()
            for idx_block in range(len(detailed_grads.items())):
                self._plot_curves(idx_alg, alg_name, idx_block, detailed_grads[idx_block]['params'], len(detailed_grads))

        self.fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        
    def _plot_curves(
        self,
        idx_alg: int,
        alg_name: str,
        idx_block: int,
        grads: Dict[str, Tensor],
        num_blocks: int
    ) -> None:
        """
        Same as parent class
        """
        
        grads = {k: v for k, v in grads.items() if k.split(".")[-1] == 'weight' and k.split(".")[0] == "synapses"}
        if idx_alg == 0:
            numel = len(grads)
            for ax in self.axes[numel:, self.axes.shape[1] - 1 - idx_block]: ax.remove()

        for idx_param, (param_name, param_grads) in enumerate(grads.items()):
            ax = self.axes[idx_param][self.axes.shape[1] - 1 - idx_block]
            self._plot_single_curves(
                idx_alg,
                ax,
                param_grads,
                alg_name,
                idx_block,
                param_name,
                num_samples=4,
                color=self.colors['block{}.'.format(idx_block) + param_name]
            )
            ax.set_ylabel('{}'.format(self.counter + idx_param + 1))
            ax.yaxis.label.set_color(self.colors['block{}.'.format(idx_block) + param_name]) 
            if idx_param == len(grads) - 1:
                start = (num_blocks - 1 - idx_block) * 20
                ax.set_xticks((0, 10, 20), (start, start + 10, start + 20))
                if idx_block == num_blocks - 1:
                    ax.set_xlabel('Backward time')
            elif idx_param == 0:
                ax.set_title('Block {}'.format(idx_block + 1), color = self.colors['block{}.'.format(idx_block) + param_name])
                ax.set_xticks([])
            else:
                ax.set_xticks([])
            ax.set_yticks([])
        self.counter += len(grads)

    def _plot_single_curves(
        self,
        idx_alg: int,
        ax,
        grads: Tensor,
        alg_name: str,
        idx_block: int,
        param_name: str,
        linewidth: int = 4,
        num_samples: int = 9,
        color: Optional[np.ndarray] = None
    ) -> None:
        """
        Same as parent class
        """
        
        grads = grads.view(grads.size(0), -1)

        if idx_alg == 0:
            self.idx_dims[(idx_block, param_name)] = [np.random.randint(grads.size(1)) for _ in range(num_samples)]
        
        for idx_sample, idx_dim in enumerate(self.idx_dims[idx_block, param_name]):
            ax.plot(grads[:, idx_dim].detach().cpu().numpy(), 
                    color='black' if alg_name == 'ep' else color, 
                    linewidth = 1.5 if alg_name == 'ep' else linewidth,
                    linestyle = self.LINESTYLES[alg_name],
                    alpha = 0.8
                    )

    def plot_bars(
        self,
        detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]]
    )-> None:
        """
        Bar plots of the layer-wise cosine similarity between EP and ID / BPTT gradients
        """
        
        cosine_dist = {}
        layer_idx = 1
        for idx_block in range(len(detailed_gradients['ep'].keys())):
            grads_ep = detailed_gradients['ep'][idx_block]['params']
            grads_bptt = detailed_gradients['bptt'][idx_block]['params']
            for param in grads_ep.keys():
                g_ep, g_bptt = grads_ep[param][-1, :].view(-1), grads_bptt[param][-1, :].view(-1)
                g_ep, g_bptt = g_ep.view(-1), g_bptt.view(-1)
                if param.split(".")[-1] == 'weight' and param.split(".")[0] == 'synapses':
                    cosine_dist['block{}.'.format(idx_block) + param] = (g_ep * g_bptt).sum() / (torch.sqrt(g_ep.pow(2).sum()) * torch.sqrt(g_bptt.pow(2).sum()))
                    
        colors = []
        xlabels = []
        barvalues = []
        layer_idx = 1
        
        for k, v in cosine_dist.items():
            colors += [self.colors[k]]
            barvalues += [v]
            xlabels += ['Layer {}'.format(layer_idx)]
            layer_idx += 1    
        
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(xlabels, barvalues, color=colors)
        plt.xticks(rotation=30, fontsize=11)
        plt.ylabel(r"$\cos(\measuredangle \hat{g}_\theta^{\rm EP}, \hat{g}_\theta^{\rm AD})$", fontsize=15)
        fig.tight_layout()
        plt.savefig("bars.svg")


class DetailedPaperPlotter(PaperPlotter):
    """
    Detailed counterpart of the PaperPlotter class where detailed gradients of * all * parameters
    (including batchnorm parameters, readout parameters) are carefully inspected
    
    TODO: some parameters are accounted for twice
    """
    
    def build_colormap(self, detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]]) -> None:
        """
        Same as parent class PaperPlotter
        """
        n_layers = sum([len(list(filter(lambda x: x.split(".")[-1] == 'weight' or x.split(".")[-1] == 'bias', list(g['params'].keys())))) for g in detailed_gradients['ep'].values()])
        cmap = plt.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, n_layers))
        
        # build dictionary mapping param names to colors
        self.colors = {}
        counter = 0
        for idx_block in range(len(detailed_gradients['ep'].items())):
            for param_name in detailed_gradients['ep'][idx_block]['params'].keys():
                self.colors['block{}.'.format(idx_block) + param_name] = colors[counter]
                counter += 1
                    
    def plot_curves(
        self,
        detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]],
    )-> None:
        """
        Same as parent class PaperPlotter
        """
        
        import os
        # '/opt/anaconda3/lib/python3.9/site-packages/latex'
        plt.rc('font', size=self.SMALL_SIZE)           # controls default text sizes
        plt.rc('axes', titlesize=self.BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('xtick', labelsize=self.MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('axes', labelsize=self.BIGGER_SIZE)    # fontsize of the x and y labels
        self.idx_dims = dict()

        n_layers = sum([len(list(filter(lambda x: x.split(".")[-1] == 'weight' or x.split(".")[-1] == 'bias', list(g['params'].keys())))) for g in detailed_gradients['ep'].values()])
        cmap = plt.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, n_layers))
        
        for idx_alg, (alg_name, detailed_grads) in enumerate(detailed_gradients.items()):
            self.counter = 0
            if idx_alg == 0:
                n_layers_per_block = [len(list(filter(lambda x: x.split(".")[-1] == 'weight' or x.split(".")[-1] == 'bias', list(g['params'].keys())))) for g in detailed_grads.values()]
                nrows = max(n_layers_per_block)
                ncols = len(detailed_grads)
                self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, nrows), layout="constrained")
                self.fig.tight_layout()
            for idx_block in range(len(detailed_grads.items())):
                self._plot_curves(idx_alg, alg_name, idx_block, detailed_grads[idx_block]['params'], len(detailed_grads))

        self.fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=1)
        
    def _plot_curves(
        self,
        idx_alg: int,
        alg_name: str,
        idx_block: int,
        grads: Dict[str, Tensor],
        num_blocks: int
    ):  
        """
        Same as parent class PaperPlotter
        """
        
        if idx_alg == 0:
            numel = len(grads)
            for ax in self.axes[numel:, self.axes.shape[1] - 1 - idx_block]: ax.remove()

        for idx_param, (param_name, param_grads) in enumerate(grads.items()):
            ax = self.axes[idx_param][self.axes.shape[1] - 1 - idx_block]
            self._plot_single_curves(
                idx_alg,
                ax,
                param_grads,
                alg_name,
                idx_block,
                param_name,
                num_samples=4,
                color=self.colors['block{}.'.format(idx_block) + param_name]
            )
            ax.set_ylabel('{}'.format(self.counter + idx_param + 1))
            ax.yaxis.label.set_color(self.colors['block{}.'.format(idx_block) + param_name]) 
            if idx_param == len(grads) - 1:
                start = (num_blocks - 1 - idx_block) * 20
                ax.set_xticks((0, 10, 20), (start, start + 10, start + 20))
                if idx_block == num_blocks - 1:
                    ax.set_xlabel('Backward time')
            else:
                ax.set_xticks([])

            ax.set_title('block{}.'.format(idx_block) + param_name, color = self.colors['block{}.'.format(idx_block) + param_name], fontsize=11)
            ax.set_yticks([])
        self.counter += len(grads)

    def plot_bars(
        self,
        detailed_gradients: Dict[int, Dict[str, Dict[str, Tensor]]]
    )-> None:
        """
        Same as parent class PaperPlotter
        """
        
        cosine_dist = {}
        layer_idx = 1
        for idx_block in range(len(detailed_gradients['ep'].keys())):
            grads_ep = detailed_gradients['ep'][idx_block]['params']
            grads_bptt = detailed_gradients['bptt'][idx_block]['params']
            for param in grads_ep.keys():
                g_ep, g_bptt = grads_ep[param][-1, :].view(-1), grads_bptt[param][-1, :].view(-1)
                g_ep, g_bptt = g_ep.view(-1), g_bptt.view(-1)
                cosine_dist['block{}.'.format(idx_block) + param] = (g_ep * g_bptt).sum() / (torch.sqrt(g_ep.pow(2).sum()) * torch.sqrt(g_bptt.pow(2).sum()))
                layer_idx += 1
                    
        colors = []
        xlabels = []
        barvalues = []
        layer_idx = 1
        
        for k, v in cosine_dist.items():
            colors += [self.colors[k]]
            barvalues += [v]
            xlabels += [k]
            layer_idx += 1    
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.bar(xlabels, barvalues, color=colors)
        plt.xticks(rotation=45, fontsize=9)
        plt.ylabel(r"$\cos(\measuredangle \hat{g}_\theta^{\rm EP}, \hat{g}_\theta^{\rm AD})$", fontsize=15)
        fig.tight_layout()
        plt.savefig("bars.svg")
        
def getScheduler(optimizer, cfg: DictConfig) -> LRScheduler:
    """
    Wrapper around a dictionary which returns a scheduler given a scheduler name
    """
    return _SCHEDULERS[cfg['name']](optimizer, **cfg['config'])       

def criterion(name='cross_entropy',device='cuda') -> Callable:
    """
    Wrapper around a dictionary which returns a callable criterion given a criterion name
    """
    return _CRITERIA[name](reduction='none').to(device)

def getCriterion(name='cross_entropy', reduction='none'):
    """
    Wrapper around a dictionary which returns a callable criterion given a criterion name
    """
    return _CRITERIA[name](reduction=reduction)

def getOptimizer(
    model: HopfieldChain,
    name: str = "adam",
    lr: float = 5e-5,
    block_configs: Optional[Dict] = None,
    disable_bn_learning: bool = True,
    rank: int = -1,
    config: Optional[Dict] = None
) -> Optimizer:
    """
    Function which returns an optimizer for a given model
    
    Args:
        model (HopfieldChain):
            model to be optimized
        name (str):
            name of the optimizer to be used
        lr (float):
            default learning rate
        block_configs (Dict):
            block-wise configurations (if block-wise or even layer-wise learning rates are used)
        disable_bn_learning (bool):
            specifies whether batchnorm parameters are learned or not
        rank (int):
            local rank (anticipating on future uses of the codebase for data parallelized simulations)
        config (Dict):
            optimizer configuration
        
    Returns:
        optimizer (Optimizer):
            the optimizer for the model under consideration
    """
    
    if not block_configs and not lr:
        raise KeyError("Optimizer config should either have key 'block_configs' or 'lr'")
    
    optim_params = []
    if rank > -1: model = model.module
    for id_block, block in enumerate(model.blocks):
        if block_configs:
            lrs =  block_configs[id_block]['lr']
            if len(lrs) > 1:
                assert len(lrs) == len(list(block.named_parameters())), "Not as many lrs as parameters!"
                block_lrs = lrs 
            else:
                block_lrs = [lrs] * len(list(block.named_parameters()))
        else:
            block_lrs = [lr] * len(list(block.named_parameters()))
            
        optim_params += [
            {'params': p, 'lr': block_lrs[id_syn]} 
            for id_syn, p in enumerate(block.parameters())
        ]

    if model.has_readout:
        readout_lr = block_configs[-1]['lr'][-1] if block_configs else lr
        optim_params += [{'params': model.readout.parameters(), 'lr': readout_lr}]

    Optim = _OPTIMIZERS[name]
    optimizer = Optim(optim_params, **config)

    def func_(module: nn.Module):
        if type(module) == nn.BatchNorm2d:
                for p in module.parameters(): p.requires_grad = False
    
    if disable_bn_learning: model.apply(func_)

    return optimizer

def load(
    model: HopfieldChain, optimizer: Optimizer, scheduler: LRScheduler, load_dir: str
)-> Tuple[HopfieldChain, Optimizer, LRScheduler, int, float]:
    """
    Loads state dicts into a model, optimizer and scheduler and returns these once updated
    
    Args:
        model (HopfieldChain):
            model at use
        optimizer (Optimizer):
            optimizer at use
        scheduler (LRScheduler):
            scheduler at use
        load_dir (str):
            where the last saved model can be found
        
    Returns:
        model (HopfieldChain):
            updated model
        optimizer (Optimizer):
            updated optimizer 
        scheduler (LRScheduler):
            updated scheduler
        start_epoch (int):
            index of the last epoch
        best (int):
            best validation performance so far
    """
    
    checkpoint = torch.load(load_dir)
    for idx, block in enumerate(model.blocks):
        block.load_state_dict(checkpoint['block'+ str(idx) + '_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if model.has_readout:
        model.readout.load_state_dict(checkpoint['readout'])
    start_epoch = checkpoint['epoch']
    best = checkpoint['top1_val'][-1]

    return model, optimizer, scheduler, start_epoch, best

def save(
    model: HopfieldChain,
    optimizer: Optimizer,
    scheduler,
    # scheduler: Optional[LRScheduler] = None,
    epoch: int,
    top1_train: List[float],
    top5_train: List[float],
    top1_val: List[float],
    top5_val: List[float],
    wct: float    
    ) -> None:
    """
    Saves model, optimizer and scheduler
    """
    
    save_dir = os.getcwd()          # With Hydra 1.1, the current directory is the * results directory * created by Hydra
    save_dict = {}
    
    for idx, block in enumerate(model.blocks):
        save_dict['block' + str(idx) + '_state_dict'] = block.state_dict()
        
    if model.has_readout:
        save_dict['readout'] = model.readout.state_dict()
        
    save_dict['optimizer_state_dict']= optimizer.state_dict()
    if scheduler is not None: save_dict['scheduler_state_dict']= scheduler.state_dict()
    save_dict['epoch'] =  epoch
    save_dict['top1_train'], save_dict['top5_train'] = top1_train, top5_train
    save_dict['top1_val'], save_dict['top5_val'] = top1_val, top5_val
    save_dict['wct'] = wct 

    torch.save(save_dict, save_dir + '/checkpoint.pth')


def setup(rank: int, world_size: int) -> None:
    """
    Sets up an environment for a data parallelized experiment
    
    Args:
        rank (int):
            local rank
        world_size (int):
            number of devices at use
    """
    
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12356"

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def train(
    dataloader: DataLoader,
    model: nn.Module,
    algorithm: Algorithm,
    criterion: Callable,
    device: device,
    optimizer: Optimizer,
    scheduler,
    rank: int,
    epoch: int
) -> Tuple[float, float, float]:
    """
    Runs a complete epoch of training
    
    Args:
        dataloader (DataLoader):
            training dataloader
        model (nn.Module):
            model at use
        algorithm (Algorithm):
            gradient computation algorithm at use (i.e. EP or AD)
        criterion (Callable):
            loss function
        device (torch.device):
            torch device 
        optimizer (Optimize):
            optimizer at use
        scheduler (LRScheduler):
            scheduler at use
        rank (int):
            local rank
        epoch (int):
            epoch index
        
    Returns:
        top1 (float):
            top1 training accuracy
        top5 (float):
            top5 training accuracy
        loss (float):
            training loss
    """
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    pbar = tqdm(dataloader, desc="Epoch {}".format(epoch + 1))

    for id_batch, (images, labels) in enumerate(pbar):
        # TO BE REMOVED LATER
        # if id_batch == 5: break
        images, labels = images.to(device), labels.to(device)
        output, loss = algorithm.run(model, images, labels, criterion, rank=rank)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        pbar.set_postfix(
            {
            "loss": "{:.3f}".format(losses.avg),
            "top1" : "{:.3f}".format(top1.avg),
            "top5" : "{:.3f}".format(top5.avg)
            }
        )

        optimizer.step()
        assert all([p.grad is not None for n, p in model.named_parameters() if p.requires_grad]), "All parameters requiring a gradient should be populated with gradients"

    if scheduler is not None: scheduler.step()

    if rank > -1:
        top1.all_reduce()
        top5.all_reduce()

    if rank in (0, -1):        
        print("Training: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (losses.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, losses.avg

def validate(
    dataloader: DataLoader,
    model: nn.Module,
    criterion,
    device: device,
    rank: int,
) -> Tuple[float, float]:
    """
    Validation loop
    
    Args:
        dataloader (DataLoader):
            validation dataloader
        model (nn.Module):
            model at use
        criterion (Callable):
            loss function
        device (torch.device):
            torch device at use
        rank (int):
            local rank
        
    Returns:
        top1 (float):
            top1 validation accuracy
        top5 (float):
            top5 validation accuracy
        losses (float):
            validation loss
    """
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    pbar = tqdm(dataloader, desc = "Validation")

    for id_batch, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        neurons = model.initialize_neurons(images)
        output, _ = model(images, neurons)
        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        losses.update(loss.mean().item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        pbar.set_postfix(
            {
            "loss": "{:.3f}".format(losses.avg),
            "top1" : "{:.3f}".format(top1.avg),
            "top5" : "{:.3f}".format(top5.avg)
            }
        )
        if rank > -1:
            top1.all_reduce()
            top5.all_reduce()

    print("Validation: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (
        losses.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, losses.avg

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

def accuracy(output: Tensor, target: Tensor, topk: Tuple[int]=(1,)) -> Tuple[float]:
    """Computes the precision@k for the specified values of k"""

    if len(target.size()) > 1: target = torch.max(target, 1)[1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res