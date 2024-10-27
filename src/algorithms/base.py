from . import AlgorithmMeta
from typing import Callable, Tuple, List, Dict
from src.models.base import HopfieldChain
from torch import Tensor
import abc

class Algorithm(metaclass=AlgorithmMeta):
    """
    Abstract (parent) algorithm class which imposes some mandatory methods to (child) algorithm classes
    and registers all these classes inside its metaclass registry
    
    Methods
    -------
    run:
        initializes neurons, computes gradients given a model, x, y and a loss function
    _compute_gradients:
        as the name indicates, computes gradients :)
    compute_gradients: 
        wrapper around _compute_gradients to use gradient computation outside of the scope of the algorithm definition
    compute_detailed_gradients:
        computes detailed / truncated gradients * with respect to parameters and neurons *
    """

    def run(
        self,
        model: HopfieldChain,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        rank: int = -1,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Runs the algorithm
        
        Args:
            model (HopfieldChain):
                model to which the algorithm is applied
            x (Tensor):
                input tensor
            y (Tensor):
                associated labels
            loss_fun (Callable):
                loss function at use
        Returns:
            out (Tensor), loss (Tensor):
                logits and loss
        """
        
        if rank > -1: model = model.module
        neurons = model.initialize_neurons(x)
        out, neurons = model(x, neurons, **kwargs)
        loss = self._compute_gradients(out, x, y, neurons, model, loss_fn)
        return out, loss

    @abc.abstractmethod
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
        Carries out the actual gradient computation
        
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

        raise NotImplementedError

    def compute_gradients(
        self,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        neurons: List[List[Tensor]],
        model: HopfieldChain,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> None:
        """
        Wrapper around _compute_gradients with same args and return
        """

        return self._compute_gradients(out, x, y, neurons, model, loss_fn, **kwargs)

    @abc.abstractmethod
    def compute_detailed_gradients(
        self,
        out: Tensor,
        x: Tensor,
        y: Tensor,
        model: HopfieldChain,
        loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Dict[int, Dict[str, Dict[str, Tensor]]]:
        """
        Computes detailed / truncated gradients  * with respect to parameters and neurons *
        
        Args:
            Same as compute_gradients
            
        Returns:
            detailed_grads (Dict[int, Dict[str, Dict[str, Tensor]]]):
                A dictionnary of detailed gradients grouped by block index and name
        """

        raise NotImplementedError