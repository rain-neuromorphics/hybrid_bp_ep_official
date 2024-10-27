from src.helpers import copy
import torch
from src.algorithms.base import Algorithm
from typing import Dict, Any, Callable
from src.models.base import HopfieldChain, HopfieldBlock
from torch import Tensor
from typing import List, Optional, Tuple
from torch.nn.modules.loss import CrossEntropyLoss

class EP(Algorithm):
    """
    Class implementing the Equilibrium Propagation (EP) algorithm
    
    This class implements the mandatory parent methods (_compute_gradients, _compute_detailed_gradients) as well
    as new methods (_compute_block_gradients, _compute_detailed_block_gradients, _compute_block_param_gradients,
    _compute_detailed_block_param_gradients)
    
    Attributes
    ----------
    beta (float):
        nudging value
    T2 (int):
        (maximal) number of steps through equilibrium being differentiated through
    
    Methods
    -------
    run:
        same as parent class (Algorithm), except that we always explicitly cache inputs (cache_x: bool = True)
    _compute_gradients:
        same as parent class (Algorithm)
    _compute_block_gradients:
        computes gradients inside a given block * with respect to parameters and neurons * 
    compute_detailed_gradients:
        same as parent class (Algorithm)
    _compute_detailed_block_gradients:
        computes detailed gradients inside a given block  * with respect to parameters and neurons *
    _compute_block_param_gradients:
        computes parameter gradients inside a given block
    _compute_detailed_block_param_gradients:
        computes detailed / truncated gradients * for parameters only * 
    """

    def __init__(
        self,
        beta: float = 0.2,
        T2: int = 5,
        **kwargs
    ) -> None:

        self.beta = beta
        self.T2 = T2
        """
        Instantiates a EP object
        
        Args:
            beta (float):
                nudging strength
            T2 (int):
                (maximal) number of steps through equilibrium being differentiated through
        """

    def run(
        self,
        model: HopfieldChain,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Runs the EP algorithm. Note that the sole difference w.r.t parent class (Algorithm)
        is that we always explicitly cache inputs (cache_x: bool = True)
        
        Args:
            model (HopfieldChain):
                model to which the algorithm is applied
            x (Tensor):
                input tensor
            y (Tensor):
                associated labels
            loss_fn (Callable):
                loss function at use
            
        Returns:
            out (Tensor), loss (Tensor):
                logits and loss
        """

        return super().run(model, x, y, loss_fn, cache_x=True, **kwargs)

    def _compute_gradients(
        self,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        neurons: List[Tensor],
        model: HopfieldChain,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> Tensor:
        """
        Carries out the actual EP gradient computation
        
        Args:
            out (Tensor):
                the model logits
            x (Tensor):
                input tensor
            y (Tensor):
                associated label
            neurons (List[List[Tensor]]):
                list of blocks, with each block being a list of tensors
            model (HopfieldChain):
                model to which the algorithm is applied
            loss_fn (Callable):
                loss function at use
        Returns:
            loss (Tensor):
                value of the loss function on x and y
        """
        
        model.zero_grad()
        assert all([p.grad is None or p.grad.equal(torch.zeros_like(p)) for p in model.parameters()]), "All model parameters should have None or zero gradients before backward pass starts!"
        for i, (block, neuron) in enumerate(zip(reversed(model.blocks), reversed(neurons))):
            
            error_x = self._compute_block_gradients(
                block,
                block.block_x,
                neuron,
                loss_fn = loss_fn if i == 0 else None,
                readout = model.readout if i == 0 else None,
                error_current = error_x if i > 0 else None,
                y = y if i == 0 else None,
                tol = False
            )

        assert all([p.grad is not None for p in model.parameters() if p.requires_grad]), "All parameters requiring a gradient should be populated gradients (EP)"

        with torch.no_grad():
            loss = loss_fn(out, y) if type(loss_fn) == CrossEntropyLoss else loss_fn(out, y.float()).sum(1)
            loss = loss.mean()
        
        return loss

    def _compute_block_gradients(
        self,
        block: HopfieldBlock,
        block_x: Tensor,
        ref_neurons: List[Tensor],
        betas: Optional[List[float]] = None,
        **kwargs
    ) -> Tensor:
        """
        Computes EP gradients inside a given block
        
        Args:
            block (HopfieldBlock):
                block under consideration
            block_x (Tensor):
                input to the block under consideration
            ref_neurons (List[Tensor]):
                free state of the block under consideration
            betas: (Optional[List[float]]):
                the nudging values being used for the EP learning rule
        
        Returns:
            error_x (Tensor):
                error signal to be passed to upstream / preceding block      
        """

        equilibria = []
        
        if betas is None: betas = (self.beta, -self.beta)

        for beta in betas:
            neurons = copy(ref_neurons)
            equilibria += [block(block_x, neurons, self.T2, beta=beta, backwards=True, **kwargs)]

        equilibria = [ref_neurons] + equilibria
        error_x = self._compute_block_param_gradients(block, block_x, equilibria, **kwargs)

        return error_x

    def compute_detailed_gradients(
        self,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        neurons: List[List[Tensor]],
        model: HopfieldChain,
        loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Dict[int, Dict[str, Dict[str, Tensor]]]:
        """
        Computes detailed / truncated EP gradients * with respect to parameters and neurons *
        
        Args:
            Same as compute_gradients
            
        Returns:
            detailed_grads (Dict[int, Dict[str, Dict[str, Tensor]]]):
                A dictionnary of detailed gradients grouped by block index and parameter name
        """
        
        model.zero_grad()
        assert all([p.grad is None or p.grad.equal(torch.zeros_like(p)) for p in model.parameters()]), "All model parameters should have None or zero gradients before backward pass starts!"
        
        detailed_grads = dict()
        for i, (block, neuron) in enumerate(zip(reversed(model.blocks), reversed(neurons))):
            error_x, detailed_grads_tmp = self._compute_detailed_block_gradients(
                block,
                block.block_x,
                neuron,
                loss_fn = loss_fn if i == 0 else None,
                readout = model.readout if i == 0 else None,
                error_current = error_x if i > 0 else None,
                y = y if i == 0 else None,
            )
            detailed_grads[len(model.blocks) - 1 - i] = detailed_grads_tmp
        
        return detailed_grads


    def _compute_detailed_block_gradients(
        self,
        block: HopfieldBlock,
        block_x: Tensor,
        ref_neurons: List[Tensor],
        betas: Optional[List[float]] = None,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:
        """
        Computes detailed / truncated EP gradients * with respect to parameters and neurons * and * inside a given block * 
        
        Args:
            block (HopfieldBlock):
                block under consideration
            block_x (Tensor):
                input to the block under consideration
            ref_neurons (List[Tensor]):
                free states of the neurons of the block under consideration
            betas (Optional[List[float]]):
                nudging values at use to apply the EP learning rule
            
        Returns:
            detailed_grads (Dict[int, Dict[str, Dict[str, Tensor]]]):
                A dictionnary of detailed gradients grouped by name for the block under consideration
        """
        
        if betas is None: betas = (-self.beta, self.beta)

        t = []

        for beta in betas:
            neurons = copy(ref_neurons)
            t += [block.compute_trajectory(block_x, neurons, self.T2, beta = beta, backwards=True, **kwargs)[-1]]

        neuron_detailed_grads = {
            'layer.{}'.format(ind): torch.vstack([
                -(j[ind] - i[ind]).unsqueeze(0) / ((betas[1] - betas[0]) * block_x.size(0))
                for (i, j) in zip(t[0][1:], t[1][1:])
            ])
            for ind in range(len(t[0][0]))
        }

        param_detailed_grads = [
            self._compute_detailed_block_param_gradients(block, block_x, [ref_neurons, i, j], betas=betas, **kwargs)
        for (i, j) in zip(t[0], t[1])
        ]
        
        param_detailed_grads = {k: torch.vstack([v[k].unsqueeze(0) for v in param_detailed_grads]) for k in param_detailed_grads[0].keys()}

        input_detailed_grads = {
            'inputs':
                (1 / block_x.size(0)) * torch.vstack([
                 self._compute_block_param_gradients(block, block_x, [ref_neurons, i, j], betas=betas, **kwargs).unsqueeze(0)
                for (i, j) in zip(t[0], t[1])
                ])
        }

        detailed_grads = {
            'neurons': neuron_detailed_grads,
            'params': param_detailed_grads
        }

        detailed_grads['neurons'].update(input_detailed_grads)
        return block_x.size(0) * input_detailed_grads['inputs'][-1, :], detailed_grads

    def _compute_block_param_gradients(
        self,
        block: HopfieldBlock,
        block_x: Tensor,
        equilibria: List[List[Tensor]],
        betas: Optional[List[float]] = None,
        **kwargs
    ) -> Tensor:
        """
        Computes EP parameter gradients * for a given block *
        
        Args:
            block (HopfieldBlock):
                block under consideration
            block_x (Tensor):
                input to the block under consideration
            equilibria (List[List[Tensor]]):
                different (nudged) steady states at use for all the layers of the block under consideration
            betas (Optional[List[float]]):
                the nudging strength at use for the EP learning rule
          
        Returns:
            error_x (Tensor):
                error current used to nudge the previous block
        """

        if betas is None: betas = (self.beta, -self.beta)

        assert all([v.is_leaf for n in equilibria for v in n]), "All neurons used inside the EP learning rule must be leaf variables"

        phis = [block.Phi(block_x, state, beta=b, **kwargs).mean() for state, b in zip(equilibria[1:], betas)]

        delta_phi = - 1 / ((betas[1] - betas[0])) * (phis[1] - phis[0])

        block.zero_grad()
        if kwargs['readout'] is not None:
            kwargs['readout'].zero_grad()
        block_x.grad = torch.zeros_like(block_x)
        delta_phi.backward()
        error_x = - block_x.grad * block_x.size(0)

        assert all([p.grad is not None for p in block.parameters() if p.requires_grad]), "All block parameters requiring a gradient should be populated with gradients (EP)"
        assert error_x is not None, "Error current is None!"
        return error_x

    def _compute_detailed_block_param_gradients(
        self,
        block: HopfieldBlock,
        block_x: Tensor,
        equilibria: List[List[Tensor]],
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Detailed / truncated counterpart of _compute_block_param_gradients
        
        Args:
            Same as _compute_block_param_gradients
            
        Returns:
            A dictionary of detailed parameter EP gradients
        """

        self._compute_block_param_gradients(block, block_x, equilibria, **kwargs)

        out = {n: p.grad.clone().detach() for n, p in block.named_parameters()}
        if kwargs['readout'] is not None:
            readout = kwargs['readout']
            out.update({'readout.' + n: p.grad.clone().detach() for n, p in readout.named_parameters()})
        return out