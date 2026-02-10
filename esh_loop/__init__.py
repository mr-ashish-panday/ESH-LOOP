"""
ESH-Loop: Adaptive Compute via Entropy-Gated Pondering
======================================================
A hybrid SSM-Attention architecture that dynamically adjusts
computational depth per-token using entropy-based routing.
"""

from .model import ESHLoopModel, ESHLoopConfig
from .layers import ESHLoopBlock
from .router import EntropyRouter

__version__ = "0.1.0"
__all__ = ["ESHLoopModel", "ESHLoopConfig", "ESHLoopBlock", "EntropyRouter"]
