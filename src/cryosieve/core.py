import argparse
import sys
from .logger import configure_logging, logger

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'CryoSieve core')
    parser.add_argument('--i',               type = str,   required = True, help = 'input star file path')
    parser.add_argument('--o',               type = str,   required = True, help = 'output star file path')
    parser.add_argument('--directory',       type = str,                    help = 'directory of particles')
    parser.add_argument('--angpix',          type = float,                  help = 'pixelsize in Angstrom')
    parser.add_argument('--volume',          type = str,   required = True, action = 'append', help = 'list of volume file paths')
    parser.add_argument('--mask',            type = str,                    help = 'mask file path')
    parser.add_argument('--retention_ratio', type = float, required = True, help = 'fraction of retained particles')
    parser.add_argument('--frequency',       type = float, required = True, help = 'cut-off highpass frequency')
    parser.add_argument('--num_gpus',        type = int,   default  = 1,    help = 'number of GPUs to execute the cryosieve program, 1 by default')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def process(args, ctx = None):
    import numpy as np
    from pathlib import Path
    from .ParticleDataset import ParticleDataset
    from .utility import mrcread
    from .sieve import barrier, sieve

    # Initialize.
    from cupy.cuda.runtime import getDeviceCount, setDevice
    if ctx is not None and ctx.distributed:
        setDevice(ctx.local_rank)
    else:
        if args.num_gpus < 1:
            raise ValueError('`--num_gpus` should be positive')
        if args.num_gpus > getDeviceCount():
            raise ValueError(f'`--num_gpus` is {args.num_gpus}, but only {getDeviceCount()} CUDA device(s) are available')

    # Input.
    dataset     = ParticleDataset(args.i, args.directory, args.angpix)
    angpix      = dataset.pixel_size
    volumes     = [np.asarray(mrcread(path), dtype = np.float64) for path in args.volume]
    mask_volume = np.asarray(mrcread(args.mask), dtype = np.float64)
    volumes     = [volume * mask_volume for volume in volumes]
    ratio       = args.retention_ratio
    threshold   = angpix / args.frequency
    output_path = Path(args.o)
    logger.info(f'Initialize ParticleDataset with given directory {str(dataset.data_dir.absolute())}')

    # Process.
    n_subset = dataset.n_random_subset()
    if n_subset != len(volumes):
        raise ValueError('Number of particle subsets should be the same as number of input volumes')

    mask = np.zeros(len(dataset), dtype = np.bool_)
    for i in range(n_subset):
        subset = dataset.get_random_subset(i + 1)
        if ctx is None or ctx.is_main:
            logger.info(f'Start sieving subset {i}, {len(subset)} particles')
        n_rem = round(ratio * len(subset))
        subset_rem = sieve(subset, volumes[i], threshold, n_rem, args.num_gpus, ctx)
        if ctx is None or ctx.is_main:
            mask[subset_rem.indices] = True
            logger.info(f'Finish sieving subset {i}, {n_rem} particles remained')

    if ctx is None or ctx.is_main:
        dataset_rem = dataset.subset(mask)
        dataset_sie = dataset.subset(~mask)
        dataset_rem.save(output_path)
        dataset_sie.save(output_path.with_stem(output_path.stem + '_sieved'))
    barrier(ctx)

def main():
    configure_logging(source='cryosieve-core')
    args = parse_arguments()

    from .utility import check_cupy
    check_cupy()

    from time import time
    from .sieve import destroy_distributed, init_distributed
    ctx = None
    time0 = time()
    try:
        ctx = init_distributed()
        process(args, ctx)
    except Exception:
        logger.exception('CryoSieve core failed during distributed startup or execution')
        raise
    finally:
        if ctx is not None:
            destroy_distributed()
    time1 = time()
    if ctx.is_main:
        logger.info(f'Execute cryosieve-core successfully in {time1 - time0:.2f}s')

if __name__ == '__main__':
    main()
