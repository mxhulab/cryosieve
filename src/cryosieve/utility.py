import mrcfile
import subprocess
import numpy as np
from time import time
from typing import Optional
from . import logger

def check_cupy():
    try:
        import cupy as cp
    except ModuleNotFoundError:
        logger.error('cannot find CuPy module')
        exit(1)
    except ImportError:
        logger.error('cannot import CuPy module, please check your CUDA environment or GPU card')
        exit(1)

def mrcread(fpath, i_slc : Optional[int] = None, cached_mrc_handles : Optional[dict] = None) -> np.memmap:
    if cached_mrc_handles is not None and fpath in cached_mrc_handles:
        mrc = cached_mrc_handles[fpath]
    else:
        mrc = mrcfile.mmap(fpath, permissive = True, mode = 'r')

    if i_slc is None:
        data = mrc.data
    elif mrc.data.ndim == 2:
        logger.warning(f"{str(fpath)} is an image, rendering it as a stack of 1 image")
        data = mrc.data
    else:
        data = mrc.data[i_slc]

    if cached_mrc_handles is not None:
        cached_mrc_handles[fpath] = mrc
    else:
        mrc.close()

    return data

def run_commands(commands, jobname = '', stdout = None, cwd = None):
    time0 = time()
    if isinstance(commands, str): commands = [commands]
    processes = [subprocess.Popen(command, shell = True, stdout = stdout, cwd = cwd) for command in commands]
    for process in processes: process.wait()
    time1 = time()
    if all(process.returncode == 0 for process in processes):
        logger.info(f'Execute {jobname} successfully in {time1 - time0:.2f}s')
    else:
        returncodes = [process.returncode for process in processes]
        logger.warning(f'Execute {jobname} ununsuccessfully! Return code(s): {returncodes}')
        exit(max(returncodes))
