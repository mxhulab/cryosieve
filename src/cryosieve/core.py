import argparse
import os
import torch
import numpy as np
import cupy as cp

from .ParticleDataset import ParticleDataset
from .utility import mrcread
from .sieve import sieve

def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM.')
    parser.add_argument('--i',               type = str,   required = True,  help = 'input star file path.')
    parser.add_argument('--o',               type = str,   required = True,  help = 'output star file path.')
    parser.add_argument('--directory',       type = str,   default  = './',  help = 'directory of particles, current directory by default.')
    parser.add_argument('--angpix',          type = float, required = True,  help = 'pixelsize in Angstrom.')
    parser.add_argument('--volume',          type = str,   required = True,  action = 'append', help = 'list of volume file paths.')
    parser.add_argument('--mask',            type = str,   required = False, help = 'mask file path.')
    parser.add_argument('--retention-ratio', type = float, default  = 0.8,   help = 'fraction of retained particles.')
    parser.add_argument('--frequency',       type = float, required = True,  help = 'cut-off highpass frequency.')
    parser.add_argument('--balance',         action = 'store_true',          help = 'make retained particles in different subsets in same size.')
    return parser.parse_args()

def main():
    # Initialize.
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank       = int(os.environ['RANK']) if is_distributed else 0
    world_size = int(os.environ['WORLD_SIZE']) if is_distributed else 1
    if is_distributed: torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(rank)
    cp.cuda.runtime.setDevice(rank)

    # Input.
    args        = parse_argument()
    dataset     = ParticleDataset(args.i, args.directory, args.angpix)
    volumes     = [mrcread(path) for path in args.volume]
    mask = mrcread(args.mask)
    volumes = [volume * mask for volume in volumes]
    ratio       = args.retention_ratio
    threshold   = args.angpix / args.frequency
    output_path = args.o

    # Process.
    n_subset = dataset.subsets.max()
    if n_subset != len(volumes):
        raise ValueError('Number of particle subsets should be the same as number of input volumes.')

    rem_number = []
    for i_subset in range(n_subset):
        dataset.reset(i_subset + 1)
        rem_number.append(round(len(dataset) * ratio))
    dataset.reset()
    if args.balance:
        min_rem_number = min(rem_number)
        rem_number = [min_rem_number for _ in rem_number]

    mask = np.zeros(len(dataset), dtype = np.bool_)
    for i_subset in range(n_subset):
        dataset.reset(i_subset + 1)
        if rank == 0:
            print(f'Processing subset {i_subset}.')
            print(f'There are {len(dataset)} particles.')
        dataset_rem = sieve(dataset, volumes[i_subset], threshold, rem_number[i_subset], rank, world_size)
        if rank == 0:
            mask[dataset_rem.indices] = True
            print(f'{len(dataset_rem)} particles remained.\n')

    if rank == 0:
        dataset.reset()
        dataset_rem, dataset_sie = dataset.split(mask)
        root, ext = os.path.splitext(output_path)
        dataset_rem.save(output_path)
        dataset_sie.save(root + "_sieved" + ext)

if __name__ == '__main__':
    main()
