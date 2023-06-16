__all__ = [
    'cupy_to_torch',
    'torch_to_cupy',
    'mrcread'
]

import cupy as cp
import numpy as np
import mrcfile
from torch.utils.dlpack import to_dlpack, from_dlpack

def cupy_to_torch(x):
    return from_dlpack(x.toDlpack())

def torch_to_cupy(x):
    return cp.fromDlpack(to_dlpack(x))

def mrcread(fpath : str, iSlc = None):
    with mrcfile.mmap(fpath, permissive = True, mode = 'r') as mrc:
        data = mrc.data if iSlc is None or mrc.data.ndim == 2 else mrc.data[iSlc]
        return np.array(data, dtype = np.float64)
