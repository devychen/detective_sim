from ._archive.base_agent import DetectiveAgent
from ._archive.holmes_agent import create_holmes_agent
from ._archive.poirot_agent import create_poirot_agent
from ._archive.marple_agent import create_marple_agent

__all__ = [
    "DetectiveAgent",
    "create_holmes_agent",
    "create_poirot_agent",
    "create_marple_agent",
]
