from .base_agent import DetectiveAgent
from .holmes_agent import create_holmes_agent
from .poirot_agent import create_poirot_agent
from .marple_agent import create_marple_agent

__all__ = [
    "DetectiveAgent",
    "create_holmes_agent",
    "create_poirot_agent",
    "create_marple_agent",
]
