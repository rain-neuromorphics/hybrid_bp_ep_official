import torch
from torch import Tensor
from torch.nn import MSELoss
import torch.nn.functional as F
from typing import List
    
def copy(neurons: List[Tensor], keep_graph=False):
    """
    Function which copies a list of Tensors
    """
    
    if isinstance(neurons, list):
        copy_ = [copy(n, keep_graph) for n in neurons]
    else:
        if keep_graph:
            copy_ = neurons.clone()
        else:
            copy_ = torch.empty_like(neurons).copy_(neurons.data).requires_grad_()

    return copy_

class NormalizedMSELoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 0.5 * F.mse_loss(input, target, reduction=self.reduction)
