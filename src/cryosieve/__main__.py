import argparse
import sys
from argparse import Namespace
from pathlib import Path
from .logger import configure_logging, emit_progress, logger


def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM')
    parser.add_argument('--i',                    type = str,   required = True,  help = 'input star file path')
    parser.add_argument('--o',                    type = str,   required = True,  help = 'output directory')
    parser.add_argument('--directory',            type = str,   required = False, help = 'directory of particles')
    parser.add_argument('--angpix',               type = float, required = False, help = 'pixelsize in Angstrom')
    parser.add_argument('--sym',                  type = str,   default  = 'C1',  help = 'molecular symmetry, C1 by default')
    parser.add_argument('--num_iters',            type = int,   default  = 10,    help = 'number of iterations for applying CryoSieve, 10 by default')
    parser.add_argument('--frequency_start',      type = float, default  = 50.,   help = 'starting threshold frquency, in Angstrom, 50 by default')
    parser.add_argument('--frequency_end',        type = float, default  = 3.,    help = 'ending threshold frquency, in Angstrom, 3 by default')
    parser.add_argument('--retention_ratio',      type = float, default  = 0.8,   help = 'fraction of retained particles in each iteration, 0.8 by default')
    parser.add_argument('--mask',                 type = str,   required = True,  help = 'mask file path')
    parser.add_argument('--random_seed',          type = int,   default  = None,  help = 'seed for postprocess phase randomization, unset by default')
    parser.add_argument('--balance',              action = 'store_true',          help = 'randomly drop particles to make all subset into the same size')
    parser.add_argument('--skip_gridding',        action = 'store_true',          help = 'skip iterative gridding correction during reconstruction')
    parser.add_argument('--iter',                 type = int,   default  = 10,    help = 'number of reconstruction gridding iterations, 10 by default')
    parser.add_argument('--num_gpus',             type = int,   default  = 1,     help = 'number of gpus to execute CryoSieve core program, 1 by default')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()


def main():
    configure_logging(source='cryosieve')
    args = parse_argument()

    from .utility import check_cupy
    check_cupy()

    import numpy as np
    from .ParticleDataset import ParticleDataset
    from .postprocess import postprocess
    from .reconstruct import reconstruct_to_mrc

    try:
        src = Path(args.i)
        if not src.exists():
            raise FileNotFoundError(f'{args.i} not found')
        elif src.suffix != '.star':
            raise ValueError(f'{args.i} is not a star file')

        dst = Path(args.o).absolute()
        dst.mkdir(parents = True, exist_ok = True)
        if not dst.is_dir():
            raise ValueError(f'{args.o} is not a directory or cannot be created')
        emit_progress(
            'cryosieve',
            'setup',
            0,
            args.num_iters,
            'iteration',
            metadata={'iteration': 0, 'totalIterations': args.num_iters},
        )

        dataset = ParticleDataset(src, args.directory, args.angpix)
        args.angpix = dataset.pixel_size
        data_dir = dataset.data_dir.absolute()
        logger.info(f'Initialize ParticleDataset with given directory {str(data_dir)}')
        if args.balance:
            dataset.balance()
        dataset.save(dst / 'iter0.star')

        # go.
        frequences = 1 / np.linspace(1.0 / args.frequency_start, 1.0 / args.frequency_end, args.num_iters)
        overall_retention_ratio = 1.0
        for i in range(args.num_iters):
            logger.info(f'Start iteration {i}, overall retaining ratio {overall_retention_ratio * 100:.2f}%, threshold frequency {frequences[i]:.2f} Angstrom')

            # reconstruct.
            for subset in (1, 2):
                logger.info(f'Start 3D reconstruction iteration {i}, subset {subset}')
                reconstruct_to_mrc(
                    dst / f'iter{i}.star',
                    dst / f'iter{i}_half{subset}.mrc',
                    directory = data_dir,
                    pixel_size = args.angpix,
                    symmetry = args.sym,
                    subset = subset,
                    apply_ctf = True,
                    skip_gridding = args.skip_gridding,
                    gridding_iterations = args.iter,
                )

            # postprocess.
            logger.info(f'Start postprocess (iteration {i})')
            result = postprocess(
                dst / f'iter{i}_half1.mrc',
                dst / f'iter{i}_half2.mrc',
                dst / f'iter{i}_pp',
                angpix = args.angpix,
                mask_path = args.mask,
                random_seed = args.random_seed,
                log_path = dst / 'postprocess.log',
            )
            logger.info(
                f'Postprocess iteration {i} successfully: '
                f'resolution {result.final_resolution:.3f} Angstrom, '
                f'B-factor {result.bfactor:.3f}'
            )

            # sieve.
            from .core import process as process_core
            core_args = Namespace(
                i = str(dst / f'iter{i}.star'),
                o = str(dst / f'iter{i + 1}.star'),
                directory = str(data_dir) if args.directory is not None else None,
                angpix = args.angpix,
                volume = [str(dst / f'iter{i}_half1.mrc'), str(dst / f'iter{i}_half2.mrc')],
                mask = args.mask,
                retention_ratio = args.retention_ratio,
                frequency = float(f'{frequences[i]:.3f}'),
                num_gpus = args.num_gpus,
            )
            process_core(core_args)
            overall_retention_ratio *= args.retention_ratio
            emit_progress(
                'cryosieve',
                'iteration-output-written',
                i + 1,
                args.num_iters,
                'iteration',
                metadata={
                    'iteration': i + 1,
                    'totalIterations': args.num_iters,
                    'retentionRatio': overall_retention_ratio,
                    'frequencyAngstrom': float(frequences[i]),
                },
                checkpoint={
                    'iteration': i + 1,
                    'total': args.num_iters,
                    'starPath': str(dst / f'iter{i + 1}.star'),
                },
            )

        logger.info('Execute CryoSieve successfully')
    except Exception:
        logger.exception('CryoSieve failed during execution')
        raise


if __name__ == '__main__':
    main()
