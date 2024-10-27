from typing import Dict, Any

class AlgorithmMeta(type):
    """
    Metaclass to register Algorithm classes inside a dictionary mapping
    algorithm name to associated algorithm class:
    
    AlgorithmMeta.REGISTRY = {"ep": EP, "bptt": BPTT}
    """
    
    REGISTRY = {}
    def __new__(cls, name, bases, attrs):
        new_cls = super(AlgorithmMeta, cls).__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

def getAlgorithm(name: str, cfg: Dict[str, Any]):
    return AlgorithmMeta.REGISTRY[name](**cfg)


from .bptt import BPTT
from .ep import EP
from .base import Algorithm


