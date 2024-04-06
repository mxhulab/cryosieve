import argparse
import csv
import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from time import sleep
from random import uniform

def parse_argument():
    parser = argparse.ArgumentParser(description = 'cryosieve-csrefine: automatic SPA 3D-refinement by calling CryoSPARC.')
    parser.add_argument('--i',         type = str, nargs    = '+',  help = 'input star file(s) or txt file(s) containing a list of star files.')
    parser.add_argument('--directory', type = str, default  = '',   help = 'directory of particles, empty (current directory) by default.')
    parser.add_argument('--o',         type = str,                  help = 'output summary csv file path. If not provided, no summary is written.')
    parser.add_argument('--sym',       type = str, default  = 'C1', help = 'molecular symmetry, C1 by default.')
    parser.add_argument('--ref',       type = str,                  help = 'initial reference model. If not provided, CryoSPARC\'s ab-initio job will be used.')
    parser.add_argument('--repeat',    type = int, default  = 1,    help = 'number of trials, 1 by default.')
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

def load_cryosparc(args):
    from cryosparc_compute.client import CommandClient
    global host, port, client, user_id, project_uid, workspace_uid, lane, lock

    host = os.environ['CRYOSPARC_MASTER_HOSTNAME']
    port = os.environ['CRYOSPARC_COMMAND_CORE_PORT']
    client = CommandClient(host = host, port = port)
    user_id = client.get_id_by_email(args.user)
    project_uid = args.project
    workspace_uid = args.workspace
    lane = args.lane
    lock = Lock()

def enqueue_and_wait(**kwargs):
    with lock:
        job_id = client.make_job(
            user_id = user_id,
            project_uid = project_uid,
            workspace_uid = workspace_uid,
            **kwargs
        )
        client.enqueue_job(project_uid, job_id, lane, user_id)

    while True:
        sleep(uniform(30, 60))
        with lock: job_status = client.get_job_status(project_uid, job_id)
        if job_status == 'completed':
            return job_id
        elif job_status in ['failed', 'killed']:
            raise RuntimeError(f'Job {job_id} is {job_status}.')

def refine(particle_meta_path, particle_blob_dir):
    import_job_id = enqueue_and_wait(
        job_type = 'import_particles',
        params = {
            'particle_meta_path' : particle_meta_path,
            'particle_blob_path' : particle_blob_dir
        }
    )

    if not import_ref_job_id:
        abinit_job_id = enqueue_and_wait(
            job_type = 'homo_abinit',
            input_group_connects = {
                'particles': f'{import_job_id}.imported_particles'
            }
        )

    refine_job_id = enqueue_and_wait(
        job_type = 'nonuniform_refine_new' if args.nu else 'homo_refine_new',
        params = {
            'refine_symmetry' : args.sym,
            'refine_gs_resplit' : args.resplit
        },
        input_group_connects = {
            'particles' : f'{import_job_id}.imported_particles',
            'volume' : f'{import_ref_job_id}.imported_volume_1' if import_ref_job_id else f'{abinit_job_id}.volume_class_0'
        }
    )

    with lock: streamlog = client.get_job_streamlog(project_uid, refine_job_id)
    text_list = [entry['text'].strip() for entry in streamlog if 'text' in entry]
    resolution = bfactor = None
    for text in text_list:
        if text.startswith('Using Filter Radius'):
            resolution = re.search(r'\(([\d.]+)A\)', text).group(1)
        elif text.startswith('Estimated Bfactor:'):
            bfactor = re.search(r'(-[\d.]+)', text).group(1)
        elif text.startswith('Split A has'):
            num_A = int(re.search(r'\b(\d+)\b', text).group(1))
        elif text.startswith('Split B has'):
            num_B = int(re.search(r'\b(\d+)\b', text).group(1))
    return [particle_meta_path, num_A + num_B, resolution, bfactor]

def main():
    args = parse_argument()
    input_path = ' '.join(f'"{file}"' for file in args.i)
    command = f'eval $(cryosparcm env) && python {Path(__file__).absolute()} ' + \
              f'--i {input_path} ' + \
              (f'--directory "{args.directory}" ' if args.directory else '') + \
              (f'--o "{args.o}" ' if args.o is not None else '') + \
              f'--sym {args.sym} ' + \
              (f'--ref "{args.ref}" ' if args.ref is not None else '') + \
              f'--repeat {args.repeat} ' + \
              f'--user {args.user} ' + \
              f'--project {args.project} ' + \
              f'--workspace {args.workspace} ' + \
              f'--lane {args.lane} ' + \
              ('--nu ' if args.nu else '') + \
              ('--resplit ' if args.resplit else '') + \
              (f'--workers {args.workers} ' if args.workers is not None else '')
    subprocess.run(command, shell = True)

def parse_meta_paths(paths):
    particle_meta_paths = []
    for path in paths:
        path = Path(path)
        if path.suffix == '.star':
            particle_meta_paths.append(str(path.absolute()))
        elif path.suffix == '.txt':
            with open(path) as fin: lines = fin.readlines()
            for line in lines:
                meta_path = Path(line.strip())
                if meta_path.suffix != '.star':
                    raise RuntimeError(f'{meta_path} is not a star file.')
                particle_meta_paths.append(str(meta_path.absolute()))
        else:
            raise RuntimeError(f'"{path}" is not a star or txt file.')
    return particle_meta_paths

if __name__ == '__main__':
    args = parse_argument()
    load_cryosparc(args)

    # Check args.
    particle_meta_paths = parse_meta_paths(args.i)
    if not Path(args.directory).is_dir():
        raise FileNotFoundError(f'"{args.directory}" is not a directory.')
    particle_blob_dir = str(Path(args.directory).absolute())
    csvfile = open(args.o, 'w')
    csvwriter = csv.writer(csvfile)

    # If ref is given, import it.
    if args.ref is not None:
        if not Path(args.ref).is_file():
            raise FileNotFoundError(f'"{args.ref}" not exists.')
        import_ref_job_id = enqueue_and_wait(
            job_type = 'import_volumes',
            params = {
                'volume_blob_path' : str(Path(args.ref).absolute())
            }
        )
    else:
        import_ref_job_id = ''

    # Call CryoSPARC's refinement.
    pool = ThreadPoolExecutor() if args.workers is None else ThreadPoolExecutor(max_workers = args.workers)
    futures = [pool.submit(refine, particle_meta_path, particle_blob_dir) for particle_meta_path, _ in product(particle_meta_paths, range(args.repeat))]
    results = [future.result() for future in futures]
    pool.shutdown()
    if any(result is None for result in results):
        raise RuntimeError('Error occurred during CryoSPARC refinement.')

    # Save results.
    csvwriter.writerow(['filename', 'number of particles', 'resolution (A)', 'b-factor (A^2)'])
    for result in results: csvwriter.writerow(result)
    csvfile.close()
