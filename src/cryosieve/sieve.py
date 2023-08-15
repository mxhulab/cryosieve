import cupy as cp
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from .kernels import convolute_ctf, highpass2d, project, translate
from .utility import cupy_to_torch, torch_to_cupy

def sieve(dataset, volume, threshold, number, rank, world_size):
    m = len(dataset)
    batch_size = 50

    # Take rank-th part of dataset.
    l, r = round(rank / world_size * m), round((rank + 1) / world_size * m)
    sampler = SequentialSampler(range(l, r))
    loader = DataLoader(dataset, batch_size, num_workers = 4, pin_memory = True, sampler = sampler)
    g = torch.zeros(m, dtype = torch.float64, device = 'cuda')
    volume = cp.array(volume, dtype = cp.float64)

    for i_batch, batch in enumerate(loader):

        # Prepare batch data.
        imgs  = torch_to_cupy(batch[0].to('cuda'))
        paras = batch[1].numpy()
        trans = paras[:, 0:2]
        quats = paras[:, 2:6]
        ctfs  = paras[:, 6:14]

        # Compute score.
        imgs = translate(imgs, trans)
        projs = convolute_ctf(project(volume, quats), ctfs) - imgs
        imgs = highpass2d(imgs, threshold)
        projs = highpass2d(projs, threshold)
        lft = l + i_batch * batch_size
        g[lft : lft + len(imgs)] = cupy_to_torch(cp.linalg.norm(projs, axis = (1, 2)) ** 2 - \
                                                 cp.linalg.norm(imgs,  axis = (1, 2)) ** 2)

    if world_size > 1: dist.reduce(g, dst = 0)
    g = g.to('cpu').numpy()
    indices = np.argsort(g)
    return dataset.subset(indices[:number])
