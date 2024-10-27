from src.helpers import copy
from src.models.base import HopfieldChain, HopfieldBlock
from typing import Dict, Callable, List
from src.algorithms.base import Algorithm
from torch import Tensor
import torch
from torch.nn.modules.loss import CrossEntropyLoss

class BPTT(Algorithm):
    """
    Class implementing Backpropagation Through Time (BPTT), also referred to as Automatic Differentiation (AD)
    or Implicit Differentiation (ID) in our paper
    
    This class implements the mandatory parent methods (_compute_gradients, _compute_detailed_gradients) as well
    as new methods (_compute_detailed_block_gradients)
    
    Attributes
    ----------
    T2 (int):
        number of steps which are run through equilibrium and differentiated through
    
    Methods
    -------
    
    _compute_gradients:
        same as parent class (Algorithm)
    compute_detailed_gradients:
        same as parent class (Algorithm)
    _compute_detailed_block_gradients:
        computes detailed gradients inside a given block  * with respect to neurons and parameters * 
    """
    
    def __init__(
        self,
        T2: int = 5,
        **kwargs
    ) -> None:
        """
        Instantiates a BPTT object
        
        Args:
            T2 (int):
                (maximal) number of steps through equilibrium being differentiated through
        """

        self.T2 = T2

    def _compute_gradients(
        self,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        neurons: List[List[Tensor]],
        model: HopfieldChain,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> Tensor:
        """
        Carries out the actual BPTT gradient computation
        
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
        
        out, _ = model(x, neurons, self.T2, use_autograd=True, tol=False, **kwargs)
        loss = loss_fn(out, y) if type(loss_fn) == CrossEntropyLoss else loss_fn(out, y.to(model._device).float()).sum(1)
        loss = loss.mean()
        model.zero_grad()
        loss.backward()

        assert all([p.grad is not None for p in model.parameters() if p.requires_grad]), "All parameters requiring a gradient should be populated with gradients (BPTT)"

        return loss

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
        Computes detailed / truncated BPTT gradients * with respect to parameters and neurons *
        
        Args:
            Same as compute_gradients
            
        Returns:
            detailed_grads (Dict[int, Dict[str, Dict[str, Tensor]]]):
                A dictionnary of detailed gradients grouped by block index and parameter name
        """
        
        assert all([n.is_leaf for b in neurons for n in b])

        detailed_grads = dict()

        for idx_block in reversed(range(len(model.blocks))):
            detailed_grads_tmp = self._compute_detailed_block_gradients(y, idx_block, model, neurons, loss_fn)
            detailed_grads[idx_block] = detailed_grads_tmp

        return detailed_grads
    
    def _compute_detailed_block_gradients(
        self,
        y: Tensor,
        idx_block: int,
        model: HopfieldBlock,
        ref_neurons: List[List[Tensor]],
        loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Computes detailed / truncated gradients * inside a given block * and * with respect to parameters and neurons * 
        
        Args:
            y (Tensor):
                label associated to currently processed inputs x
            idx_block (int):
                index of the block under consideration
            model (HopfieldBlock):
                block under consideration (not the whole HopfieldChain!)
            ref_neurons (List[List[Tensor]]):
                free state of all the neurons (all layers from all blocks)
            loss_fn (Callable):
                loss function at use
            
        Returns:
            detailed_grads (Dict[str, Dict[str, Tensor]]):
                A dictionnary of detailed gradients grouped by name for the block under consideration
        """
        
        detailed_grads = {
            'params': [],
            'inputs': [],
            'neurons': []
        }

        for t in range(0, self.T2 + 1):
            # Initialize neurons properly
            neurons = copy(ref_neurons, keep_graph=False)
            leaf_neurons = [[n for n in b] for b in neurons]
            assert all([n.grad is None for b in leaf_neurons for n in b])

            # Fetch block input
            h = model.blocks[idx_block].block_x.detach()
            h.requires_grad = True
            leaf_input = h

            # Execute the downstream block dynamics
            for i, (b, n) in enumerate(zip(model.blocks[idx_block:], neurons[idx_block:])):
                num_iterations = t if i == 0 else self.T2
                neurons[i] = b(h, n, num_iterations, retain_grad=True, use_autograd=True)
                h = neurons[i][-1]
            
            # Compute loss
            if model.has_readout: h = model.readout(h)
            loss = loss_fn(h, y) if type(loss_fn) == CrossEntropyLoss else loss_fn(h, y.to(model._device).float()).sum(1)
            loss = loss.mean()
            model.zero_grad()
            loss.backward()

            # Retrieve gradients
            detailed_grads['neurons'] += [
                [torch.zeros_like(n) if n.grad is None else n.grad for n in leaf_neurons[idx_block]]
            # for b in leaf_neurons
            ]

            detailed_grads['inputs'] += [
                torch.zeros_like(leaf_input) if leaf_input.grad is None else leaf_input.grad
            ]

            tmp = {k: torch.zeros_like(p) if p.grad is None else p.grad for k, p in model.blocks[idx_block].named_parameters()}
            if idx_block == len(model.blocks) - 1 and model.has_readout:
                tmp.update({"readout." + k: torch.zeros_like(p) if p.grad is None else p.grad for k, p in model.readout.named_parameters()})
            detailed_grads['params'] += [tmp]
    
        detailed_grads_tmp = {
                    'params': {
                        n: torch.vstack([v[n].unsqueeze(0) for v in detailed_grads['params']])
                    for n in detailed_grads['params'][0].keys()
                    },
                    'neurons': {
                        'layer.{}'.format(i) :
                        torch.cumsum(
                            torch.vstack([n[i].unsqueeze(0) for n in detailed_grads['neurons']]),
                        dim=0)
                        for i in range(len(detailed_grads['neurons'][0]))
                    }
        }
        detailed_grads_tmp['neurons'].update({'inputs': torch.vstack([v.unsqueeze(0) for v in detailed_grads['inputs']])})
        
        return detailed_grads_tmp


        
