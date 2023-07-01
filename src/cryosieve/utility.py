import cupy as cp
import numpy as np
import mrcfile
import subprocess
from time import time
from torch.utils.dlpack import to_dlpack, from_dlpack

def cupy_to_torch(x):
    return from_dlpack(x.toDlpack())

def torch_to_cupy(x):
    return cp.fromDlpack(to_dlpack(x))

def mrcread(fpath : str, iSlc = None):
    with mrcfile.mmap(fpath, permissive = True, mode = 'r') as mrc:
        data = mrc.data if iSlc is None or mrc.data.ndim == 2 else mrc.data[iSlc]
        return np.array(data, dtype = np.float64)

def run_commands(commands, msg = '', stdout = None):
    time0 = time()
    if isinstance(commands, str): commands = [commands]
    processes = [subprocess.Popen(command, shell = True, stdout = stdout) for command in commands]
    for process in processes: process.wait()
    time1 = time()
    if all(process.returncode == 0 for process in processes):
        print(f'[Subprocesses completed successfully in {time1 - time0:.2f}s.][{msg}]')
    else:
        print(f'[Subprocesses reported an error.',
              f'Return code: {[process.returncode for process in processes]}.][{msg}]')
        exit()
