import argparse
import sys
from . import logger

def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM')
    parser.add_argument('--reconstruct_software', type = str,   required = True,  help = 'command for reconstruction')
    parser.add_argument('--postprocess_software', type = str,   required = False, help = 'command for postprocessing')
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
    parser.add_argument('--balance',              action = 'store_true',          help = 'randomly drop particles to make all subset into the same size')
    parser.add_argument('--num_gpus',             type = int,   default  = 1,     help = 'number of gpus to execute CryoSieve core program, 1 by default')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    args = parse_argument()
    if args.postprocess_software is not None:
        logger.warning('Argument `--postprocess_software` will be deprecated')

    from .utility import check_cupy
    check_cupy()

    import numpy as np
    from pathlib import Path
    from .ParticleDataset import ParticleDataset
    from .utility import run_commands

    src = Path(args.i)
    if not src.exists():
        raise FileNotFoundError(f'{args.i} not found')
    elif src.suffix != '.star':
        raise ValueError(f'{args.i} is not a star file')

    dst = Path(args.o).absolute()
    dst.mkdir(parents = True, exist_ok = True)
    if not dst.is_dir():
        raise ValueError(f'{args.o} is not a directory or cannot be created')

    dataset = ParticleDataset(src, args.directory, args.angpix)
    data_dir = dataset.data_dir.absolute()
    logger.info(f'Initialize ParticleDataset with given directory {str(data_dir)}')
    if args.balance: dataset.balance()
    dataset.save(dst / 'iter0.star')

    # go.
    frequences = 1 / np.linspace(1.0 / args.frequency_start, 1.0 / args.frequency_end, args.num_iters)
    overall_retention_ratio = 1.0
    for i in range(args.num_iters):
        logger.info(f'Start iteration {i}, overall retaining ratio {overall_retention_ratio * 100:.2f}%, threshold frequency {frequences[i]:.2f} Angstrom')

        # reconstruct.
        commands = [
            ' '.join([
                args.reconstruct_software,
                f'--i "{str(dst / f"iter{i}.star")}"',
                f'--o "{str(dst / f"iter{i}_half1.mrc")}"',
                f'--angpix {args.angpix}',
                f'--sym {args.sym}',
                '--ctf true',
                '--subset 1',
                f'>"{str(dst / f"iter{i}_reconstruct_half1.txt")}"',
            ]),
            ' '.join([
                args.reconstruct_software,
                f'--i "{str(dst / f"iter{i}.star")}"',
                f'--o "{str(dst / f"iter{i}_half2.mrc")}"',
                f'--angpix {args.angpix}',
                f'--sym {args.sym}',
                '--ctf true',
                '--subset 2',
                f'>"{str(dst / f"iter{i}_reconstruct_half2.txt")}"',
            ])
        ]
        run_commands(commands, f'3D-reconstruction (iteration {i})', cwd = data_dir)

        # postprocess.
        if args.postprocess_software is not None:
            pp_dir = dst / f'postprocess_iter{i}'
            pp_dir.mkdir(parents = True, exist_ok = True)
            command = ' '.join([
                args.postprocess_software,
                f'--mask "{args.mask}"',
                f'--i "{str(dst / f"iter{i}_half1.mrc")}"',
                f'--i2 "{str(dst / f"iter{i}_half2.mrc")}"',
                f'--o "{str(pp_dir / f"iter{i}")}"',
                f'--angpix {args.angpix}',
                '--auto_bfac',
                '--autob_lowres 10',
                f'>"{str(dst / f"postprocess_iter{i}.txt")}"',
            ])
            run_commands(command, f'postprocess (iteration {i})')

        # sieve.
        command = ' '.join([
            'cryosieve-core',
            f'--i "{str(dst / f"iter{i}.star")}"',
            f'--o "{str(dst / f"iter{i + 1}.star")}"',
            f'--directory "{str(data_dir)}"' if args.directory is not None else '',
            f'--angpix {args.angpix}',
            f'--volume "{str(dst / f"iter{i}_half1.mrc")}"',
            f'--volume "{str(dst / f"iter{i}_half2.mrc")}"',
            f'--mask {args.mask}',
            f'--retention_ratio {args.retention_ratio}',
            f'--frequency {frequences[i]:.3f}',
            f'--num_gpus {args.num_gpus}',
        ])
        run_commands(command, f'sieve (iteration {i})')
        overall_retention_ratio *= args.retention_ratio

    logger.info('Execute CryoSieve successfully')

if __name__ == '__main__':
    main()
