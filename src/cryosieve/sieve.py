import cupy as cp
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.distributed import reduce
from .kernels import convolute_ctf, highpass2d, project, translate

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return torch.from_numpy(imgs), torch.from_numpy(labels)

def cupy_to_torch(x):
    return from_dlpack(x.toDlpack())

def torch_to_cupy(x):
    return cp.fromDlpack(to_dlpack(x))

def sieve(dataset, volume, threshold, number, rank, world_size):
    m = len(dataset)
    batch_size = 50

    # Take rank-th part of dataset.
    l, r = round(rank / world_size * m), round((rank + 1) / world_size * m)
    subset = Subset(dataset, list(range(l, r)))
    loader = DataLoader(subset, batch_size, collate_fn = collate_fn)
    g = torch.zeros(m, dtype = torch.float64, device = 'cuda')
    volume = cp.asarray(volume, dtype = cp.float64)

    for i_batch, batch in enumerate(loader):

        # Prepare batch data.
        imgs = torch_to_cupy(batch[0].to('cuda', dtype = torch.float64, non_blocking = True, copy = True))
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
        rgt = lft + len(imgs)
        g[lft : rgt] = cupy_to_torch(cp.linalg.norm(projs, axis = (1, 2)) ** 2 - cp.linalg.norm(imgs, axis = (1, 2)) ** 2)

    if world_size > 1: reduce(g, dst = 0)
    g = g.to('cpu').numpy()
    indices = np.argsort(g)
    mask = np.zeros(m, dtype = np.bool_)
    mask[indices[:number]] = True
    return dataset.subset(mask)
