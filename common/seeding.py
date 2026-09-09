import random
import numpy as np
import torch

torch.set_float32_matmul_precision('medium')

def set_seeds(seed=27):
    """
    Set seeds for reproducibility across random, numpy, and torch libraries.
    
    Args:
        seed (int): Seed value to use for all random number generators. Default is 27.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)