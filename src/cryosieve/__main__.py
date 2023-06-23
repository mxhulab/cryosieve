import argparse
import shutil
import subprocess
import torch
import numpy as np
from pathlib import Path
from time import time

def parse_argument():
    parser = argparse.ArgumentParser(description = 'CryoSieve: beta.')
    parser.add_argument('--reconstruct_software', type = str,   required = True,  help = 'path of the software for reconstruction.')
    parser.add_argument('--postprocess_software', type = str,   required = False, help = 'path of the software for postporcess.')
    parser.add_argument('--num_procs',            type = int,   default  = 1,     help = 'number of processes to execute the reconstruction program.')
    parser.add_argument('--num_gpus',             type = int,   default  = torch.cuda.device_count(), help = 'number of gpus to execute the cryosieve program.')
    parser.add_argument('--i',                    type = str,   required = True,  help = 'input star file path.')
    parser.add_argument('--o',                    type = str,   required = True,  help = 'output path prefix.')
    parser.add_argument('--directory',            type = str,   default  = './',  help = 'directory of particles, empty (current directory) by default.')
    parser.add_argument('--angpix',               type = float, required = True,  help = 'pixelsize in Angstrom.')
    parser.add_argument('--sym',                  type = str,   default  = 'c1',  help = 'molecular symmetry.')
    parser.add_argument('--num_iters',            type = int,   default  = 10,    help = 'number of iterations for applying CryoSieve.')
    parser.add_argument('--frequency_start',      type = float, default  = 50.,   help = 'starting threshold frquency, in Angstrom.')
    parser.add_argument('--frequency_end',        type = float, default  = 3.,    help = 'ending threshold frquency, in Angstrom.')
    parser.add_argument('--retention_ratio',      type = float, default  = 0.8,   help = 'fraction of retained particles in each iteration.')
    parser.add_argument('--mask',                 type = str,   required = True,  help = 'mask file path.')
    parser.add_argument('--balance',              action = 'store_true',          help = 'make remaining particles in different subsets in same size.')
    return parser.parse_args()

def run_command(commands, msg = '', stdout = None):
    time0 = time()
    processes = [subprocess.Popen(command, shell = True, stdout = stdout) for command in commands]
    for process in processes: process.wait()
    time1 = time()
    if all(process.returncode == 0 for process in processes):
        print(f'[Subprocesses completed successfully in {time1 - time0:.2f}s.][{msg}]')
    else:
        print(f'[Subprocesses reported an error.',
              f'Return code: {[process.returncode for process in processes]}.][{msg}]')
        exit()

def main():
    args = parse_argument()

    # copy the star file to output directory as iteration 0.
    src = Path(f'{args.i}')
    dst = Path(f'{args.o}_iter0.star')
    if not src.exists():
        raise FileNotFoundError(f'{args.i} not found.')
    elif src.suffix != '.star':
        raise ValueError(f'{args.i} is not a star file.')
    dst.absolute().parent.mkdir(parents = True, exist_ok = True)
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass

    # go.
    frequences = 1 / np.linspace(1.0 / args.frequency_start, 1.0 / args.frequency_end, args.num_iters)
    overall_retention_ratio = 1.0
    for i in range(args.num_iters):
        print(f'[ITER {i}][Overall Retaining ratio: {overall_retention_ratio * 100:.2f}%][Threshold frequency : {frequences[i]:.2f}A]')

        # reconstruct.
        commands = [f'mpirun -n {args.num_procs} {args.reconstruct_software} --i {args.o}_iter{i}.star --o {args.o}_iter{i}_half1.mrc --angpix {args.angpix} --sym {args.sym} --ctf true --subset 1 >{args.o}_iter{i}_reconstruct_half1.txt',
                    f'mpirun -n {args.num_procs} {args.reconstruct_software} --i {args.o}_iter{i}.star --o {args.o}_iter{i}_half2.mrc --angpix {args.angpix} --sym {args.sym} --ctf true --subset 2 >{args.o}_iter{i}_reconstruct_half2.txt']
        run_command(commands, f'RECONSTRUCT_DONE, ITERATION {i}')

        # postprocess.
        if args.postprocess_software is not None:
            Path(f'{args.o}_postprocess_iter{i}').mkdir(parents = True, exist_ok = True)
            run_command([f'{args.postprocess_software} --mask {args.mask} --i {args.o}_iter{i}_half1.mrc --i2 {args.o}_iter{i}_half2.mrc --o {args.o}_postprocess_iter{i}/iter{i} --angpix {args.angpix} --auto_bfac --autob_lowres 10 >{args.o}_postprocess_iter{i}.txt'], f'POSTPROCESS_DONE, ITERATION {i}')

        # sieve.
        run_command([f'torchrun --standalone --nnodes=1 --nproc_per_node={args.num_gpus} -m cryosieve.core --i {args.o}_iter{i}.star --o {args.o}_iter{i + 1}.star --directory {args.directory} --angpix {args.angpix} --volume {args.o}_iter{i}_half1.mrc --volume {args.o}_iter{i}_half2.mrc --mask {args.mask} --retention-ratio {args.retention_ratio} --frequency {frequences[i]:.2f} {"--balance" if args.balance else ""} >{args.o}_iter{i}_sieve.txt'], f'SIEVE_DONE, ITERATION {i}')
        overall_retention_ratio *= args.retention_ratio

    print('EXECUTION IN SUCCESS')

if __name__ == '__main__':
    main()
