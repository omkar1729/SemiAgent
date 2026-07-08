"""Shared deterministic-seed utility.

Import and call `set_all_seeds()` as the first executable line of every
training and evaluation script so results are reproducible.
"""
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG that affects this project and force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
