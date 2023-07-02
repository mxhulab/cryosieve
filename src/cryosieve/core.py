import argparse
import os
import torch
import cupy as cp
import numpy as np
from time import time

from .ParticleDataset import ParticleDataset
from .utility import mrcread, run_commands
from .sieve import sieve

def core_parser():
    parser = argparse.ArgumentParser(description = 'CryoSieve core.')
    parser.add_argument('--i',               type = str,   required = True,  help = 'input star file path.')
    parser.add_argument('--o',               type = str,   required = True,  help = 'output star file path.')
    parser.add_argument('--directory',       type = str,   default  = '',    help = 'directory of particles, empty (current directory) by default.')
    parser.add_argument('--angpix',          type = float, required = True,  help = 'pixelsize in Angstrom.')
    parser.add_argument('--volume',          type = str,   required = True,  action = 'append', help = 'list of volume file paths.')
    parser.add_argument('--mask',            type = str,   required = False, help = 'mask file path.')
    parser.add_argument('--retention_ratio', type = float, default  = 0.8,   help = 'fraction of retained particles, 0.8 by default.')
    parser.add_argument('--frequency',       type = float, required = True,  help = 'cut-off highpass frequency.')
    parser.add_argument('--balance',         action = 'store_true',          help = 'make retained particles in different subsets in same size.')
    return parser

def main(args):
    # Initialize.
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank       = int(os.environ['RANK']) if is_distributed else 0
    world_size = int(os.environ['WORLD_SIZE']) if is_distributed else 1
    if is_distributed: torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(rank)
    cp.cuda.runtime.setDevice(rank)

    # Input.
    dataset     = ParticleDataset(args.i, args.directory, args.angpix)
    volumes     = [mrcread(path) for path in args.volume]
    mask        = mrcread(args.mask)
    volumes     = [volume * mask for volume in volumes]
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

def core():
    parser = core_parser()
    parser.add_argument('--num_gpus',        type = int,   default  = 1,     help = 'number of GPUs to execute the cryosieve program, 1 by default.')
    args = parser.parse_args()

    if args.num_gpus == 1:
        time0 = time()
        main(args)
        time1 = time()
        print(f'[Processes completed successfully in {time1 - time0:.2f}s.][SINGLE-GPU CRYOSIEVE-CORE DONE.]')
    else:
        core_args = f'--i {args.i} --o {args.o} {f"--directory {args.directory}" if args.directory else ""} --angpix {args.angpix} {" ".join([f"--volume {volume}" for volume in args.volume])} {f"--mask {args.mask}" if args.mask is not None else ""} --retention_ratio {args.retention_ratio} --frequency {args.frequency} {"--balance" if args.balance else ""}'
        run_commands(f'torchrun --standalone --nnodes=1 --nproc_per_node={args.num_gpus} -m cryosieve.core ' + core_args, 'MULTI-GPU CRYOSIEVE-CORE DONE.')

if __name__ == '__main__':
    main(core_parser().parse_args())
