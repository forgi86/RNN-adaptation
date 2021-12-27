"""
This file defines the datatypes of variables to be used globally.
"""
import torch
import numpy as np
from functools import partial

mat = np.atleast_2d
tensor = partial(torch.tensor, dtype=torch.float32)
device = torch.device('cpu')



