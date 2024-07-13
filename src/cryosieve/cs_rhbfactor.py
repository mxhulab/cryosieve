import argparse
import csv
import starfile
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .cs_refine import parse_meta_paths

def parse_argument():
    parser = argparse.ArgumentParser(description = 'cryosieve-csrhbfactor: automatic Rosenthal-Henderson B-factor estimation by calling CryoSPARC.')
    parser.add_argument('--i',         type = str, nargs    = '+',  help = 'input star file(s) or txt file(s) containing a list of star files.')
    parser.add_argument('--directory', type = str, default  = '',   help = 'directory of particles, empty (current directory) by default.')
    parser.add_argument('--o',         type = str, required = True, help = 'output summary csv file path.')
    parser.add_argument('--sym',       type = str, default  = 'C1', help = 'molecular symmetry, C1 by default.')
    parser.add_argument('--ref',       type = str,                  help = 'initial reference model. If not provided, CryoSPARC\'s ab-initio job will be used.')
    parser.add_argument('--ini_high',  type = float,                help = 'initial resolution.')
    parser.add_argument('--voltage',   type = int, default  = 300,  help = 'acceleration voltage (kV), 300 by default. Only 200 and 300 supported!')
    parser.add_argument('--repeat',    type = int, default  = 1,    help = 'number of trials, 1 by default.')
    parser.add_argument('--halves',    type = int, default  = 4,    help = 'number of times executing halvings, 4 by default.')
    parser.add_argument('--user',      type = str, required = True, help = 'e-mail address of the user of CryoSPARC.')
    parser.add_argument('--project',   type = str, required = True, help = 'project UID in CryoSPARC.')
    parser.add_argument('--workspace', type = str, required = True, help = 'workspace UID in CryoSPARC.')
    parser.add_argument('--lane',      type = str, required = True, help = 'lane selected for computing in CryoSPARC.')
    parser.add_argument('--nu',        action = 'store_true',       help = 'use non-uniform refinement.')
    parser.add_argument('--resplit',   action = 'store_true',       help = 'force re-do GS split.')
    parser.add_argument('--workers',   type = int,                  help = 'number of workers to run CryoSPARC job, unlimited by default.')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def main():
    args = parse_argument()

    # Check args.
    particle_meta_paths = parse_meta_paths(args.i)
    if args.voltage not in [200, 300]:
        raise ValueError('Only 200 or 300kV acceleration voltage supported.')
    if len(args.sym) == 0 or args.sym[0] not in ['C', 'D', 'T', 'O', 'I']:
        raise ValueError('Invalid molecular symmetry.')
    if args.sym[0] == 'C':
        N = int(args.sym[1:])
    elif args.sym[0] == 'D':
        N = 2 * int(args.sym[1:])
    elif args.sym[0] == 'T':
        N = 12
    elif args.sym[0] == 'O':
        N = 24
    elif args.sym[0] == 'I':
        N = 60
    else:
        raise ValueError('Invalid molecular symmetry.')
    csvfile = open(args.o, 'w')
    csvwriter = csv.writer(csvfile)

    # For each star file, take random halves and save it to a temporary file.
    tmp_dir = tempfile.TemporaryDirectory()
    with open(f'{tmp_dir.name}/list.txt', 'w') as lst_file:
        for i, meta_path in enumerate(particle_meta_paths):
            star = starfile.read(meta_path, always_dict = True)
            if len(star) == 1 and (0 in star or '' in star or 'images' in star):
                optics = None
                particles = star[0] if 0 in star else star[''] if '' in star else star['images']
            elif len(star) == 2 and ('optics' in star and 'particles' in star):
                optics = star['optics']
                particles = star['particles']
            else:
                raise ValueError('Invalid particle star file.')

            for j in range(args.repeat):
                for k in range(args.halves + 1):
                    particles_j = particles.sample(frac = 0.5 ** k) if args.resplit else \
                                  pd.concat([particles[particles['rlnRandomSubset'] == 1].sample(frac = 0.5 ** k),
                                             particles[particles['rlnRandomSubset'] == 2].sample(frac = 0.5 ** k)])
                    tmp_file = f'{tmp_dir.name}/{i}_{j}_{k}.star'
                    if optics is None:
                        starfile.write({'images' : particles_j}, tmp_file, overwrite = True)
                    else:
                        starfile.write({'optics' : optics, 'particles' : particles_j}, tmp_file, overwrite = True)
                    print(tmp_file, file = lst_file)

    # Call cryosieve-csrefine to estimate resolutions.
    command = f'cryosieve-csrefine --i {tmp_dir.name}/list.txt ' + \
              (f'--directory "{args.directory}" ' if args.directory else '') + \
              f'--o {tmp_dir.name}/summary.csv ' + \
              f'--sym {args.sym} ' + \
              (f'--ref "{args.ref}" ' if args.ref is not None else '') + \
              (f'--ini_high {args.ini_high} ' if args.ini_high is not None else '') + \
              f'--user {args.user} ' + \
              f'--project {args.project} ' + \
              f'--workspace {args.workspace} ' + \
              f'--lane {args.lane} ' + \
              ('--nu ' if args.nu else '') + \
              ('--resplit ' if args.resplit else '') + \
              (f'--workers {args.workers} ' if args.workers is not None else '')
    process = subprocess.run(command, shell = True)
    if process.returncode:
        raise RuntimeError('Error in running cryosieve-csrefine.')

    # Fit RH-curve.
    data = pd.read_csv(f'{tmp_dir.name}/summary.csv')
    num_points = args.repeat * (args.halves + 1)
    rh_bfactor_curve = lambda x, a : np.log(1437.5695 if args.voltage else 1831.7256) + a / (2 * x ** 2) - np.log(x)

    # Save results.
    csvwriter.writerow(['filename', 'RH-b-factor (A^2)'])
    for i, meta_path in enumerate(particle_meta_paths):
        i_slice = slice(i * num_points, (i + 1) * num_points)
        lognum_particles = np.log(N * data.iloc[i_slice]['number of particles'])
        resolutions = data.iloc[i_slice]['resolution (A)']
        rh_bfactor = -curve_fit(rh_bfactor_curve, resolutions, lognum_particles)[0][0]
        csvwriter.writerow([meta_path, rh_bfactor])
    csvfile.close()
