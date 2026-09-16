import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 27):
    """Set deterministic seeds for Python, NumPy and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer reproducibility over maximum kernel speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False