import argparse
import os
import sys
from pathlib import Path
from time import time
from . import logger

def core_parser():
    parser = argparse.ArgumentParser(description = 'CryoSieve core')
    parser.add_argument('--i',               type = str,   required = True, help = 'input star file path')
    parser.add_argument('--o',               type = str,   required = True, help = 'output star file path')
    parser.add_argument('--directory',       type = str,                    help = 'directory of particles')
    parser.add_argument('--angpix',          type = float,                  help = 'pixelsize in Angstrom')
    parser.add_argument('--volume',          type = str,   required = True, action = 'append', help = 'list of volume file paths')
    parser.add_argument('--mask',            type = str,                    help = 'mask file path')
    parser.add_argument('--retention_ratio', type = float, required = True, help = 'fraction of retained particles')
    parser.add_argument('--frequency',       type = float, required = True, help = 'cut-off highpass frequency')
    return parser

def main(args):
    import torch
    import torch.distributed
    import numpy as np
    import cupy as cp
    from .ParticleDataset import ParticleDataset
    from .utility import mrcread
    from .sieve import sieve

    # Initialize.
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank       = int(os.environ['RANK']) if is_distributed else 0
    world_size = int(os.environ['WORLD_SIZE']) if is_distributed else 1
    if is_distributed:
        if torch.distributed.is_nccl_available():
            torch.distributed.init_process_group('nccl')
        elif torch.distributed.is_gloo_available():
            torch.distributed.init_process_group('gloo')
        else:
            raise RuntimeError('Need nccl or gloo for multi-GPU cryosieve-core')
    torch.cuda.set_device(rank)
    cp.cuda.runtime.setDevice(rank)

    # Input.
    dataset     = ParticleDataset(args.i, args.directory, args.angpix)
    volumes     = [np.asarray(mrcread(path), dtype = np.float64) for path in args.volume]
    mask        = np.asarray(mrcread(args.mask), dtype = np.float64)
    volumes     = [volume * mask for volume in volumes]
    ratio       = args.retention_ratio
    threshold   = args.angpix / args.frequency
    output_path = Path(args.o)
    logger.info(f'Initialize ParticleDataset with given directory {str(dataset.data_dir.absolute())}')

    # Process.
    n_subset = dataset.n_random_subset()
    if n_subset != len(volumes):
        raise ValueError('Number of particle subsets should be the same as number of input volumes')

    mask = np.zeros(len(dataset), dtype = np.bool_)
    for i in range(n_subset):
        subset = dataset.get_random_subset(i + 1)
        if rank == 0:
            logger.info(f'Start sieving subset {i}, {len(subset)} particles')
        n_rem = round(ratio * len(subset))
        subset_rem = sieve(subset, volumes[i], threshold, n_rem, rank, world_size)
        if rank == 0:
            mask[subset_rem.particles.index] = True
            logger.info(f'Finish sieving subset {i}, {n_rem} particles remained')

    if rank == 0:
        dataset_rem = dataset.subset(mask)
        dataset_sie = dataset.subset(~mask)
        dataset_rem.save(output_path)
        dataset_sie.save(output_path.with_stem(output_path.stem + '_sieved'))

def core():
    parser = core_parser()
    parser.add_argument('--num_gpus', type = int, default  = 1, help = 'number of GPUs to execute the cryosieve program, 1 by default')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    args = parser.parse_args()

    from .utility import check_cupy
    check_cupy()

    from .utility import run_commands
    if args.num_gpus == 1:
        time0 = time()
        main(args)
        time1 = time()
        logger.info(f'Execute cryosieve-core (single GPU) successfully in {time1 - time0:.2f}s')
    else:
        core_args = ' '.join([
            f'--i "{args.i}"',
            f'--o "{args.o}"',
            f'--directory "{args.directory}"' if args.directory else '',
            f'--angpix {args.angpix}',
            " ".join(f"--volume {volume}" for volume in args.volume),
            f'--mask "{args.mask}"' if args.mask is not None else '',
            f'--retention_ratio {args.retention_ratio}',
            f'--frequency {args.frequency}'
        ])
        run_commands(f'torchrun --standalone --nnodes=1 --nproc_per_node={args.num_gpus} -m cryosieve.core ' + core_args, 'cryosieve-core (multi-GPU)')

if __name__ == '__main__':
    main(core_parser().parse_args())
