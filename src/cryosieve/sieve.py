import cupy as cp
import numpy as np
from threading import Thread
from torch.utils.data import DataLoader
from .kernels import convolute_ctf, highpass2d, project, translate
from .logger import logger

def collate_fn(batch):
    imgs, paras = zip(*batch)
    imgs = np.stack(imgs)
    paras = np.stack(paras)
    return imgs, paras

def score_particles(dataset, volume, threshold, device_id, num_gpus, g):
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

    cp.cuda.runtime.setDevice(device_id)
    volume = cp.asarray(volume, dtype = cp.float64)

    for i_batch, batch in enumerate(loader):

        # Prepare batch data
        imgs = cp.asarray(batch[0], dtype = cp.float64)
        paras = batch[1]
        trans = paras[:, 0:2]
        quats = paras[:, 2:6]
        ctfs  = paras[:, 6:14]

        # Compute score
        imgs = translate(imgs, trans)
        projs = convolute_ctf(project(volume, quats), ctfs) - imgs
        imgs = highpass2d(imgs, threshold)
        projs = highpass2d(projs, threshold)
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

def sieve(dataset, volume, threshold, number, num_gpus):
    m = len(dataset)
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
