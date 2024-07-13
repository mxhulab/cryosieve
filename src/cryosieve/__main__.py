import argparse
import sys

def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: a particle sorting and sieving software for single particle analysis in cryo-EM.')
    parser.add_argument('--reconstruct_software', type = str,   required = True,  help = 'command for reconstruction.')
    parser.add_argument('--postprocess_software', type = str,   required = False, help = 'command for postprocessing.')
    parser.add_argument('--i',                    type = str,   required = True,  help = 'input star file path.')
    parser.add_argument('--o',                    type = str,   required = True,  help = 'output path prefix.')
    parser.add_argument('--angpix',               type = float, required = True,  help = 'pixelsize in Angstrom.')
    parser.add_argument('--sym',                  type = str,   default  = 'C1',  help = 'molecular symmetry, C1 by default.')
    parser.add_argument('--num_iters',            type = int,   default  = 10,    help = 'number of iterations for applying CryoSieve, 10 by default.')
    parser.add_argument('--frequency_start',      type = float, default  = 50.,   help = 'starting threshold frquency, in Angstrom, 50A by default.')
    parser.add_argument('--frequency_end',        type = float, default  = 3.,    help = 'ending threshold frquency, in Angstrom, 3A by default.')
    parser.add_argument('--retention_ratio',      type = float, default  = 0.8,   help = 'fraction of retained particles in each iteration, 0.8 by default.')
    parser.add_argument('--mask',                 type = str,   required = True,  help = 'mask file path.')
    parser.add_argument('--balance',              action = 'store_true',          help = 'randomly drop particles to make all subset into the same size.')
    parser.add_argument('--num_gpus',             type = int,   default  = 1,     help = 'number of gpus to execute CryoSieve core program, 1 by default.')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    args = parse_argument()
    if args.postprocess_software is not None:
        print('The `--postprocess_software` will be deprecated soon. CryoSieve will implement its own postprocessing.', file = sys.stderr)

    from .utility import check_cupy
    check_cupy()

    import numpy as np
    from pathlib import Path
    from .ParticleDataset import ParticleDataset
    from .utility import run_commands

    # prepare.
    src = Path(f'{args.i}')
    dst = Path(f'{args.o}_iter0.star')
    if not src.exists():
        raise FileNotFoundError(f'{args.i} not found.')
    elif src.suffix != '.star':
        raise ValueError(f'{args.i} is not a star file.')
    dst.absolute().parent.mkdir(parents = True, exist_ok = True)

    dataset = ParticleDataset(src, '', args.angpix)
    if args.balance: dataset.balance()
    dataset.save(dst)

    # go.
    frequences = 1 / np.linspace(1.0 / args.frequency_start, 1.0 / args.frequency_end, args.num_iters)
    overall_retention_ratio = 1.0
    for i in range(args.num_iters):
        print(f'[ITER {i}][Overall Retaining ratio: {overall_retention_ratio * 100:.2f}%][Threshold frequency : {frequences[i]:.2f}A]')

        # reconstruct.
        commands = [f'{args.reconstruct_software} --i {args.o}_iter{i}.star --o {args.o}_iter{i}_half1.mrc --angpix {args.angpix} --sym {args.sym} --ctf true --subset 1 >{args.o}_iter{i}_reconstruct_half1.txt',
                    f'{args.reconstruct_software} --i {args.o}_iter{i}.star --o {args.o}_iter{i}_half2.mrc --angpix {args.angpix} --sym {args.sym} --ctf true --subset 2 >{args.o}_iter{i}_reconstruct_half2.txt']
        run_commands(commands, f'RECONSTRUCT_DONE, ITERATION {i}')

        # postprocess.
        if args.postprocess_software is not None:
            Path(f'{args.o}_postprocess_iter{i}').mkdir(parents = True, exist_ok = True)
            run_commands(f'{args.postprocess_software} --mask {args.mask} --i {args.o}_iter{i}_half1.mrc --i2 {args.o}_iter{i}_half2.mrc --o {args.o}_postprocess_iter{i}/iter{i} --angpix {args.angpix} --auto_bfac --autob_lowres 10 >{args.o}_postprocess_iter{i}.txt', f'POSTPROCESS_DONE, ITERATION {i}')

        # sieve.
        run_commands(f'cryosieve-core --i {args.o}_iter{i}.star --o {args.o}_iter{i + 1}.star --angpix {args.angpix} --volume {args.o}_iter{i}_half1.mrc --volume {args.o}_iter{i}_half2.mrc --mask {args.mask} --retention_ratio {args.retention_ratio} --frequency {frequences[i]:.3f} --num_gpus {args.num_gpus} >{args.o}_iter{i}_sieve.txt', f'SIEVE_DONE, ITERATION {i}')
        overall_retention_ratio *= args.retention_ratio

    print('EXECUTION IN SUCCESS')

if __name__ == '__main__':
    main()
