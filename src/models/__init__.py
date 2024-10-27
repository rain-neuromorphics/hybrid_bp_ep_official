import torch.nn.functional as F
import torch
import math
from torch import Tensor

class ModelMeta(type):
    """"
    Metaclass registering model classes
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(ModelMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

def getModel(name: str = 'vgg', *args, **kwargs):
    return ModelMeta.REGISTRY[name](*args, **kwargs)

class HopfieldBlockMeta(type):
    """"
    Metaclass registering block classes
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(HopfieldBlockMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

class IdentityModule(torch.nn.Module):
    """
    Custom identity module
    """
    
    def __init__(self):
        super(IdentityModule,self).__init__()
        self.kernel_size = -1
        self.stride = 1
    def forward(self, inputs):
        return inputs

def getHopfieldBlockClass(name: str = 'convpool'):
    """
    Wrapper around metaclass HopfieldBlockMeta registry to get HopfieldBlock classes by their name
    """
    return HopfieldBlockMeta.REGISTRY[name]

def getWeightInit(name: str):
    """
    Wrapper around metaclass WeightInitMeta registry to get WeightInit classes by their name
    """
    return WeightInitMeta.REGISTRY[name]

def getNeuronInit(name: str):
    """
    Wrapper around metaclass NeuronInitMeta registry to get NeuronInit classes by their name
    """
    return NeuronInitMeta.REGISTRY[name]

class StepperMeta(type):
    """
    Metaclass to register stepper classes 
    """
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(StepperMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

def hard(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5

def hard_ern19(x):
    return torch.clamp(x, min=0, max=1)

def hard_lab21(x):
    return torch.clamp(0.5 * x, min=0, max=1)

def soft(x):
    return 1 / (1 + torch.exp(-4 * (x + 2)))

def silu(x):
    return x / (1 + torch.exp(-x)) 

def relu(x):
    return F.relu(x)

class CustomHardActivation:
    """
    Custom module to implement a family of hard sigmoid functions parametrized by their slope
    Examples:
        "Ernoult activation": activation = 1
        "Laborieux activation": activation = 0.5
    """
    def __init__(self, activation: int = 0.5):
        self.activation = activation
    
    def __call__(self, x: Tensor) -> Tensor:
        return torch.clamp(self.activation * x, min=0, max=1)

def clamp(**kwargs) -> CustomHardActivation:
    """
    Returns a CustomHardActivation object based on some configs specified inside **kwargs
    """
    return CustomHardActivation(**kwargs)

class NeuronInitMeta(type):
    """
    Metaclass to register NeuronInit classes
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(NeuronInitMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
        
class WeightInitMeta(type):
    """
    Metaclass to register WeightInit classes
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(WeightInitMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
    
def custom_kaiming_uniform_(tensor, alpha=1):
    """
    Custom Kaiming uniform initialization with a scaling factor alpha.
    """
    bool_conv = tensor.dim() > 2
    
    if bool_conv:
        # fan_in = tensor.size(1) if tensor.dim() > 1 else 1
        fan_in = 1
        for d in tensor.size()[1:]: fan_in *= d
    else:
        fan_in = tensor.size(1)
        
    bound = alpha * math.sqrt(1.0 / fan_in)
        
    with torch.no_grad():
        tensor.uniform_(-bound, bound)

def gaussian_symmetric_random_matrix(N: int, V: int) -> Tensor:
    """
    Returns a randomly sampled symmetric matrix
    """
    
    A = torch.randn(N, N) * torch.sqrt(torch.tensor(V / N))  # Gaussian entries with variance V/N
    A = torch.tril(A)  # lower triangular part of A
    A = A + A.t()  # symmetric part of A
    indices = torch.arange(N)
    A[indices, indices] = A[indices, indices] * torch.sqrt(torch.tensor(2.))  # increase variance of diagonal elements
    return A

def random_symmetric(shape: int, scale_off_diag: float, scale_on_diag: float) -> Tensor:
    """
    Returns a randomly sampled symmetric matrix
    """
    
    if len(shape) != 3:
        raise ValueError("Only shapes of length 3 are supported for this function.")

    # shape is expected to be (channels, size, size)
    channels, size, _ = shape

    # Initialize an empty tensor to store the symmetric matrices
    symmetric_matrix = torch.empty(shape)

    for c in range(channels):
        # generate matrix for each channel
        matrix = scale_off_diag * torch.randn((size, size))
        # Enforce symmetry
        symmetric_matrix[c] = matrix + matrix.t()

    # set the diagonal entries to have variance 2V/N
    for c in range(channels):
        indices = torch.arange(size)
        symmetric_matrix[c, indices, indices] = scale_on_diag * torch.randn(size)

    return symmetric_matrix    

from .initializers import *
from .base import *
from .vgg import *
from .resnet import *