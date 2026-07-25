import os
from dataclasses import dataclass

import cupy as cp
import numpy as np
import torch
import torch.distributed as dist
from threading import Thread
from torch.utils.data import DataLoader
from .kernels import convolute_ctf, highpass_2d, project, translate_2d
from .logger import logger


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed CryoSieve requires CUDA')
        torch.cuda.set_device(local_rank)
        cp.cuda.runtime.setDevice(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend = 'nccl')
    return DistributedContext(rank = rank, local_rank = local_rank, world_size = world_size, distributed = distributed)


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx):
    if ctx is not None and ctx.distributed and dist.is_initialized():
        dist.barrier()

def collate_fn(batch):
    imgs, paras = zip(*batch)
    imgs = np.stack(imgs)
    paras = np.stack(paras)
    return imgs, paras

def score_particles(dataset, volume, threshold, device_id, num_gpus, g, cuda_device_id = None):
    m = len(dataset)
    batch_size = 50

    # Take device_id-th part of dataset
    l, r = round(device_id / num_gpus * m), round((device_id + 1) / num_gpus * m)
    mask = np.zeros(m, dtype = np.bool_)
    mask[l : r] = True
    subset = dataset.subset(mask)
    loader = DataLoader(subset, batch_size, collate_fn = collate_fn)
    n_batch = len(loader)
    log_interval = min(max(1, (n_batch + 4) // 5), 200)
    scores = cp.empty(r - l, dtype = cp.float64)

    cp.cuda.runtime.setDevice(device_id if cuda_device_id is None else cuda_device_id)
    volume = cp.asarray(volume, dtype = cp.float64)

    for i_batch, batch in enumerate(loader):

        # Prepare batch data
        imgs = cp.asarray(batch[0], dtype = cp.float64)
        paras = batch[1]
        trans = paras[:, 0:2]
        quats = paras[:, 2:6]
        ctfs  = paras[:, 6:14]

        # Compute score
        imgs = translate_2d(imgs, trans)
        projs = convolute_ctf(project(volume, quats), ctfs) - imgs
        imgs = highpass_2d(imgs, threshold)
        projs = highpass_2d(projs, threshold)
        start = i_batch * batch_size
        stop = start + len(imgs)
        scores[start : stop] = cp.linalg.norm(projs, axis = (1, 2)) ** 2 - cp.linalg.norm(imgs, axis = (1, 2)) ** 2
        if (i_batch + 1) % log_interval == 0 or i_batch + 1 == n_batch:
            logger.info(f'[GPU {device_id}][{i_batch + 1}/{n_batch}] Scored particle batches')

    g[l : r] = cp.asnumpy(scores)

def score_particles_safe(dataset, volume, threshold, device_id, num_gpus, g, errors):
    try:
        score_particles(dataset, volume, threshold, device_id, num_gpus, g)
    except BaseException as error:
        errors[device_id] = error


def score_particles_distributed(dataset, volume, threshold, ctx):
    m = len(dataset)
    l, r = round(ctx.rank / ctx.world_size * m), round((ctx.rank + 1) / ctx.world_size * m)
    local_mask = np.zeros(m, dtype = np.bool_)
    local_mask[l : r] = True
    local_dataset = dataset.subset(local_mask)
    local_scores = np.empty(r - l, dtype = np.float64)
    score_particles(local_dataset, volume, threshold, 0, 1, local_scores, ctx.local_rank)
    scores = torch.zeros(m, device = torch.device(f'cuda:{ctx.local_rank}'), dtype = torch.float64)
    if r > l:
        scores[l : r] = torch.as_tensor(local_scores, device = scores.device, dtype = scores.dtype)
    dist.all_reduce(scores, op = dist.ReduceOp.SUM)
    return scores.cpu().numpy()


def sieve(dataset, volume, threshold, number, num_gpus, ctx = None):
    m = len(dataset)
    if ctx is not None and ctx.distributed:
        g = score_particles_distributed(dataset, volume, threshold, ctx)
    else:
        g = np.empty(m, dtype = np.float64)
        errors = [None] * num_gpus

        if num_gpus == 1:
            score_particles(dataset, volume, threshold, 0, 1, g)
        else:
            threads = [
                Thread(target = score_particles_safe, args = (dataset, volume, threshold, tid, num_gpus, g, errors))
                for tid in range(num_gpus)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for error in errors:
                if error is not None:
                    raise error

    indices = np.argsort(g)
    mask = np.zeros(m, dtype = np.bool_)
    mask[indices[:number]] = True
    return dataset.subset(mask)
