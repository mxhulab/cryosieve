import argparse
import os
import subprocess
import sys
from pathlib import Path
from threading import Lock

try:
    from .logger import logger
except ImportError:
    # Ensure running this script in $(cryosparcm env)
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from cryosieve.logger import logger

cryosparc_job_total = None
cryosparc_jobs_generated = 0
cryosparc_jobs_enqueued = 0
cryosparc_jobs_completed = 0
dry_run = False
lock = Lock()

def parse_argument():
    parser = argparse.ArgumentParser(description = 'cryosieve-csrefine: automatic SPA 3D-refinement by calling CryoSPARC')
    parser.add_argument('--i',         type = str, nargs    = '+',  help = 'input star file(s) or txt file(s) containing a list of star files')
    parser.add_argument('--directory', type = str, default  = '',   help = 'directory of particles, empty (current directory) by default')
    parser.add_argument('--o',         type = str,                  help = 'output summary csv file path. If not provided, no summary is written')
    parser.add_argument('--sym',       type = str, default  = 'C1', help = 'molecular symmetry, C1 by default')
    parser.add_argument('--ref',       type = str,                  help = 'initial reference model. If not provided, CryoSPARC\'s ab-initio job will be used')
    parser.add_argument('--ini_high',  type = float,                help = 'initial resolution')
    parser.add_argument('--repeat',    type = int, default  = 1,    help = 'number of trials, 1 by default')
    parser.add_argument('--user',      type = str, required = True, help = 'e-mail address of the user of CryoSPARC')
    parser.add_argument('--project',   type = str, required = True, help = 'project UID in CryoSPARC')
    parser.add_argument('--workspace', type = str, required = True, help = 'workspace UID in CryoSPARC')
    parser.add_argument('--lane',      type = str, required = True, help = 'lane selected for computing in CryoSPARC')
    parser.add_argument('--nu',        action = 'store_true',       help = 'use non-uniform refinement')
    parser.add_argument('--local',     action = 'store_true',       help = 'use local refinement after homogeneous / non-uniform refinement')
    parser.add_argument('--min_angular_step', type = float, default = 0.01, help = 'minimum angular step for local refinement')
    parser.add_argument('--resplit',   action = 'store_true',       help = 'force re-do GS split')
    parser.add_argument('--workers',   type = int,                  help = 'number of workers to run CryoSPARC job, unlimited by default')
    parser.add_argument('--dry_run',   action = 'store_true',       help = 'simulate CryoSPARC jobs without connecting to CryoSPARC')
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

class DryRunClient:
    def __init__(self):
        self.job_count = 0

    def make_job(self, **kwargs):
        self.job_count += 1
        return f'_J{self.job_count}'

    def enqueue_job(self, project_uid, job_id, lane, user_id):
        pass

    def get_job_status(self, project_uid, job_id):
        return 'completed'

    def get_job_streamlog(self, project_uid, job_id):
        return [
            {'text': 'Split A has 500 particles'},
            {'text': 'Split B has 500 particles'},
            {'text': 'Using Filter Radius: 100 (3.00A)'},
            {'text': 'Estimated Bfactor: -100.0'},
        ]

    def get_project_dir_abs(self, project_uid):
        return str(Path.cwd())

    def get_job_result(self, project_uid, result):
        return {'metafile': f'dry_run_{result}.cs'}

def load_dry_run(args):
    global client, user_id, project_uid, workspace_uid, lane, dry_run

    client = DryRunClient()
    user_id = args.user
    project_uid = args.project
    workspace_uid = args.workspace
    lane = args.lane
    dry_run = True
    logger.info('Dry run enabled: simulate CryoSPARC jobs without connecting to CryoSPARC')

def load_cryosparc_env():
    try:
        from cryosparc_compute.client import CommandClient  # noqa: F401
        return
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ['bash', '-lc', 'cryosparcm env'],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True
        )
    except FileNotFoundError as exc:
        raise RuntimeError('bash is required to load CryoSPARC environment') from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or '').strip()
        message = 'failed to load CryoSPARC environment with `cryosparcm env`'
        if detail:
            message += f': {detail}'
        raise RuntimeError(message) from exc

    state = parse_shell_python_state(result.stdout)
    os.environ.update(state.get('env', {}))
    for path in state.get('sys_path', []):
        if path and path not in sys.path:
            sys.path.append(path)

def parse_shell_python_state(script):
    command = 'eval "$1" && python - <<\'PY\'\nimport json\nimport os\nimport sys\nprint(json.dumps({"env": dict(os.environ), "sys_path": sys.path}))\nPY'
    result = subprocess.run(
        ['bash', '-c', command, 'cryosparcm-env', script],
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True
    )
    return __import__('json').loads(result.stdout)

def load_cryosparc(args):
    load_cryosparc_env()
    from cryosparc_compute.client import CommandClient
    global host, port, client, user_id, project_uid, workspace_uid, lane

    host = os.environ['CRYOSPARC_MASTER_HOSTNAME']
    port = os.environ['CRYOSPARC_COMMAND_CORE_PORT']
    client = CommandClient(host = host, port = port)
    user_id = client.get_id_by_email(args.user)
    project_uid = args.project
    workspace_uid = args.workspace
    lane = args.lane

def enqueue_and_wait(local_lane, **kwargs):
    from time import sleep
    from random import uniform

    global cryosparc_jobs_generated, cryosparc_jobs_enqueued, cryosparc_jobs_completed
    job_type = kwargs.get('job_type', 'unknown')

    with lock:
        job_id = client.make_job(
            user_id = user_id,
            project_uid = project_uid,
            workspace_uid = workspace_uid,
            **kwargs
        )
        cryosparc_jobs_generated += 1
        logger.info(f'[{cryosparc_jobs_generated}/{cryosparc_job_total}] Generated CryoSPARC job {job_id} ({job_type})')

        client.enqueue_job(project_uid, job_id, local_lane, user_id)
        cryosparc_jobs_enqueued += 1
        logger.info(f'[{cryosparc_jobs_enqueued}/{cryosparc_job_total}] Enqueued CryoSPARC job {job_id} ({job_type}, lane={local_lane})')

    while True:
        if not dry_run:
            sleep(uniform(20, 40))
        with lock:
            job_status = client.get_job_status(project_uid, job_id)
            if job_status == 'completed':
                cryosparc_jobs_completed += 1
                logger.info(f'[{cryosparc_jobs_completed}/{cryosparc_job_total}] Completed CryoSPARC job {job_id} ({job_type})')
                return job_id
            elif job_status in ['failed', 'killed']:
                logger.error(f'[{cryosparc_jobs_completed}/{cryosparc_job_total}] CryoSPARC job {job_id} {job_status} ({job_type})')
                raise RuntimeError(f'Job {job_id} is {job_status}')

def import_particles(particle_meta_path):
    import_job_id = enqueue_and_wait(
        None,
        job_type = 'import_particles',
        params = {
            'particle_meta_path' : particle_meta_path,
            'particle_blob_path' : str(Path(args.directory).absolute())
        }
    )
    return import_job_id

def abinit(import_particles_job_id):
    abinit_job_id = enqueue_and_wait(
        lane,
        job_type = 'homo_abinit',
        input_group_connects = {
            'particles': f'{import_particles_job_id}.imported_particles'
        }
    )
    return abinit_job_id

def refine(particle, volume):
    params = {
        'refine_symmetry' : args.sym,
        'refine_gs_resplit' : args.resplit
    }
    if args.ini_high is not None: params['refine_res_init'] = args.ini_high
    refine_job_id = enqueue_and_wait(
        lane,
        job_type = 'nonuniform_refine_new' if args.nu else 'homo_refine_new',
        params = params,
        input_group_connects = {
            'particles' : particle,
            'volume' : volume
        }
    )
    return refine_job_id

def local_refine(refine_job_id):
    local_refine_job_id = enqueue_and_wait(
        lane,
        job_type = 'new_local_refine',
        params = {
            'refine_symmetry' : args.sym,
            'use_alignment_prior' : True,
            'min_angular_step' : args.min_angular_step
        },
        input_group_connects = {
            'particles' : f'{refine_job_id}.particles',
            'volume' : f'{refine_job_id}.volume',
            'mask' : f'{refine_job_id}.mask'
        }
    )
    return local_refine_job_id

def parse_result(refine_job_id):
    import re

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

    with lock: alignment3D_path = client.get_project_dir_abs(project_uid) + '/' + client.get_job_result(project_uid, f'{refine_job_id}.particles.alignments3D')['metafile']
    return [num_A + num_B, resolution, bfactor, alignment3D_path]

def check_results(results, info = None):
    if info is not None:
        logger.info(f'{info} {results}')
    if any(result is None for result in results):
        raise RuntimeError('Error occurred during execution')

def main():
    args = parse_argument()
    input_path = ' '.join(f'"{file}"' for file in args.i)
    command = ' '.join([
        'eval $(cryosparcm env) && python',
        str(Path(__file__).absolute()),
        f'--i {input_path}',
        f'--directory "{args.directory}"' if args.directory else '',
        f'--o "{args.o}"' if args.o is not None else '',
        f'--sym {args.sym}',
        f'--ref "{args.ref}"' if args.ref is not None else '',
        f'--ini_high {args.ini_high}' if args.ini_high is not None else '',
        f'--repeat {args.repeat}',
        f'--user {args.user}',
        f'--project {args.project}',
        f'--workspace {args.workspace}',
        f'--lane {args.lane}',
        '--nu' if args.nu else '',
        '--local' if args.local else '',
        '--resplit' if args.resplit else '',
        '--dry_run' if args.dry_run else '',
        f'--workers {args.workers}' if args.workers is not None else '',
    ])
    import subprocess
    subprocess.run(command, shell = True)

def calculate_cryosparc_job_counts(num_particle_meta_paths):
    return {
        'import_particles' : num_particle_meta_paths,
        'import_volumes' : 1 if args.ref is not None else 0,
        'ab_inito' : 0 if args.ref is not None else num_particle_meta_paths,
        'refine' : num_particle_meta_paths * args.repeat,
        'local_refine' : num_particle_meta_paths * args.repeat if args.local else 0
    }

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
                    raise RuntimeError(f'{meta_path} is not a star file')
                particle_meta_paths.append(str(meta_path.absolute()))
        else:
            raise RuntimeError(f'"{path}" is not a star or txt file')
    return particle_meta_paths

if __name__ == '__main__':
    args = parse_argument()
    if args.dry_run:
        load_dry_run(args)
    else:
        load_cryosparc(args)

    # Check args.
    particle_meta_paths = parse_meta_paths(args.i)
    if not Path(args.directory).is_dir():
        raise FileNotFoundError(f'"{args.directory}" is not a directory')
    if args.ref is not None:
        if not Path(args.ref).is_file():
            raise FileNotFoundError(f'"{args.ref}" does not exist')

    cryosparc_job_counts = calculate_cryosparc_job_counts(len(particle_meta_paths))
    cryosparc_job_total = sum(cryosparc_job_counts.values())
    logger.info(
        f'Planned CryoSPARC jobs: total={cryosparc_job_total}, '
        f'import_particles={cryosparc_job_counts["import_particles"]}, '
        f'import_volumes={cryosparc_job_counts["import_volumes"]}, '
        f'ab_inito={cryosparc_job_counts["ab_inito"]}, '
        f'homo/nu_refine={cryosparc_job_counts["refine"]}, '
        f'local_refine={cryosparc_job_counts["local_refine"]}, '
        f'particle inputs={len(particle_meta_paths)}, repeat={args.repeat}, '
        f'ref={args.ref is not None}, local={args.local}'
    )

    # Setup.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor() if args.workers is None else ThreadPoolExecutor(max_workers = args.workers)

    # Import particles.
    import_particles_tasks = [pool.submit(import_particles, particle_meta_path) for particle_meta_path in particle_meta_paths]
    import_particles_job_ids = [task.result() for task in import_particles_tasks]
    check_results(import_particles_job_ids, 'Import particles:')
    particles = [f'{import_particles_job_id}.imported_particles' for import_particles_job_id in import_particles_job_ids]

    # Import volumes, if ref is given.
    if args.ref is not None:
        import_ref_job_id = enqueue_and_wait(
            None,
            job_type = 'import_volumes',
            params = {
                'volume_blob_path' : str(Path(args.ref).absolute())
            }
        )
        check_results([import_ref_job_id], 'Imported reference maps:')
        volumes = [f'{import_ref_job_id}.imported_volume_1'] * len(particles)
    # Otherwise, use ab-initio reconstruction.
    else:
        abinit_tasks = [pool.submit(abinit, import_particles_job_id) for import_particles_job_id in import_particles_job_ids]
        abinit_job_ids = [task.result() for task in abinit_tasks]
        check_results(abinit_job_ids, 'Ab-initio reconstruction:')
        volumes = [f'{abinit_job_id}.volume_class_0' for abinit_job_id in abinit_job_ids]

    # Homogenous / non-uniform refinement.
    from itertools import product
    refine_tasks = [pool.submit(refine, particle, volume) for (particle, volume), _ in product(zip(particles, volumes), range(args.repeat))]
    refine_job_ids = [task.result() for task in refine_tasks]
    check_results(refine_job_ids, 'Homogeneous / Non-uniform refinement:')

    # Local refinement.
    if args.local:
        refine_tasks = [pool.submit(local_refine, refine_job_id) for refine_job_id in refine_job_ids]
        refine_job_ids = [task.result() for task in refine_tasks]
        check_results(refine_job_ids, 'Local refinement:')

    # Get results.
    parse_result_tasks = [pool.submit(parse_result, refine_job_id) for refine_job_id in refine_job_ids]
    results = [task.result() for task in parse_result_tasks]
    check_results(results)

    # Shutdown.
    pool.shutdown()

    # Save results.
    import csv
    with open(args.o, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'number of particles', 'resolution (A)', 'b-factor (A^2)', 'alignment3D'])
        for i, result in enumerate(results):
            particle_meta_path = particle_meta_paths[i // args.repeat]
            csvwriter.writerow([particle_meta_path] + result)
