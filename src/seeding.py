"""Global seeding for reproducible runs across numpy / torch / pennylane.

PennyLane draws its randomness from numpy (and, for torch-interface circuits,
from torch), so seeding both covers the quantum models too. We also pin the
`PYTHONHASHSEED` and request deterministic cuDNN where available. Determinism on
GPU is best-effort: some CUDA kernels remain nondeterministic, so we log the
seed in the results provenance rather than promising bit-exact reproducibility.
"""
from __future__ import annotations
import os
import random


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed all RNGs the harness touches. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuDNN determinism: reproducible at the cost of some speed.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
