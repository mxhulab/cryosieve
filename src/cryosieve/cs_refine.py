import argparse
import sys
from pathlib import Path
from threading import Lock
from .logger import logger

cryosparc_job_total = None
cryosparc_jobs_generated = 0
cryosparc_jobs_enqueued = 0
cryosparc_jobs_completed = 0
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

def load_cryosparc(args):
    if args.dry_run:
        from .autocsparc import DryRunClient
        client = DryRunClient()
        logger.info('Dry run enabled: simulate CryoSPARC jobs without connecting to CryoSPARC')
        return client, args.user
    else:
        from .autocsparc import CommandClient
        client = CommandClient()
        user_id = client.get_id_by_email(args.user)
        return client, user_id

def enqueue_and_wait(session, lane, **kwargs):
    from time import sleep
    from random import uniform

    global cryosparc_jobs_generated, cryosparc_jobs_enqueued, cryosparc_jobs_completed, lock
    job_type = kwargs.get('job_type', 'unknown')
    client, user_id, project_uid, workspace_uid, _ = session
    from .autocsparc import DryRunClient
    dry_run = isinstance(client, DryRunClient)

    with lock:
        job_id = client.make_job(
            user_id = user_id,
            project_uid = project_uid,
            workspace_uid = workspace_uid,
            **kwargs
        )
        cryosparc_jobs_generated += 1
        logger.info(f'[{cryosparc_jobs_generated}/{cryosparc_job_total}] Generated CryoSPARC job {job_id} ({job_type})')

        client.enqueue_job(project_uid, job_id, lane, user_id)
        cryosparc_jobs_enqueued += 1
        logger.info(f'[{cryosparc_jobs_enqueued}/{cryosparc_job_total}] Enqueued CryoSPARC job {job_id} ({job_type}, lane={lane})')

        if dry_run:
            cryosparc_jobs_completed += 1
            logger.info(f'[{cryosparc_jobs_completed}/{cryosparc_job_total}] Dry run: skip waiting for CryoSPARC job {job_id} ({job_type})')
            return job_id

    while True:
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

def import_particles(session, meta_path, data_dir):
    import_job_id = enqueue_and_wait(
        session,
        lane = None,
        job_type = 'import_particles',
        params = {
            'particle_meta_path' : meta_path,
            'particle_blob_path' : data_dir
        }
    )
    return import_job_id

def import_volume(session, ref):
    import_ref_job_id = enqueue_and_wait(
        session,
        lane = None,
        job_type = 'import_volumes',
        params = {
            'volume_blob_path' : ref
        }
    )
    return import_ref_job_id

def abinit(session, particle):
    abinit_job_id = enqueue_and_wait(
        session,
        lane = session[4],
        job_type = 'homo_abinit',
        input_group_connects = {
            'particles': particle
        }
    )
    return abinit_job_id

def refine(session, particle, volume, ini_high, sym, resplit, nu):
    params = {
        'refine_symmetry' : sym,
        'refine_gs_resplit' : resplit
    }
    if ini_high is not None: params['refine_res_init'] = ini_high
    refine_job_id = enqueue_and_wait(
        session,
        lane = session[4],
        job_type = 'nonuniform_refine_new' if nu else 'homo_refine_new',
        params = params,
        input_group_connects = {
            'particles' : particle,
            'volume' : volume
        }
    )
    return refine_job_id

def local_refine(session, refine_job_id, sym, min_angular_step):
    local_refine_job_id = enqueue_and_wait(
        session,
        lane = session[4],
        job_type = 'new_local_refine',
        params = {
            'refine_symmetry' : sym,
            'use_alignment_prior' : True,
            'min_angular_step' : min_angular_step
        },
        input_group_connects = {
            'particles' : f'{refine_job_id}.particles',
            'volume' : f'{refine_job_id}.volume',
            'mask' : f'{refine_job_id}.mask'
        }
    )
    return local_refine_job_id

def parse_result(session, refine_job_id):
    import re
    client, _, project_uid, _, _ = session

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

def check_results(results, info = None, dry_run = False):
    if info is not None and not dry_run:
        logger.info(f'{info} {results}')
    if any(result is None for result in results):
        raise RuntimeError('Error occurred during execution')

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

def process(args):
    client, user_id = load_cryosparc(args)
    session = (client, user_id, args.project, args.workspace, args.lane)

    # Check args
    particle_meta_paths = parse_meta_paths(args.i)
    if not Path(args.directory).is_dir():
        raise FileNotFoundError(f'"{args.directory}" is not a directory')
    particle_data_dir = str(Path(args.directory).absolute())
    if args.ref is not None:
        if not Path(args.ref).is_file():
            raise FileNotFoundError(f'"{args.ref}" does not exist')
        ref = str(Path(args.ref).absolute())
    else:
        ref = None

    global cryosparc_job_total
    cryosparc_job_counts = {
        'import_particles' : len(particle_meta_paths),
        'import_volumes' : 1 if ref is not None else 0,
        'ab_inito' : 0 if ref is not None else len(particle_meta_paths),
        'refine' : len(particle_meta_paths) * args.repeat,
        'local_refine' : len(particle_meta_paths) * args.repeat if args.local else 0
    }
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

    # Setup
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor() if args.workers is None else ThreadPoolExecutor(max_workers = args.workers)

    # Import particles
    import_particles_tasks = [pool.submit(import_particles, session, particle_meta_path, particle_data_dir) for particle_meta_path in particle_meta_paths]
    import_particles_job_ids = [task.result() for task in import_particles_tasks]
    check_results(import_particles_job_ids, 'Import particles:', args.dry_run)
    particles = [f'{import_particles_job_id}.imported_particles' for import_particles_job_id in import_particles_job_ids]

    # Import volumes, if ref is given
    if ref is not None:
        import_ref_job_id = import_volume(session, ref)
        check_results([import_ref_job_id], 'Imported reference maps:', args.dry_run)
        volumes = [f'{import_ref_job_id}.imported_volume_1'] * len(particles)

    # Otherwise, use ab-initio reconstruction
    else:
        abinit_tasks = [pool.submit(abinit, session, particle) for particle in particles]
        abinit_job_ids = [task.result() for task in abinit_tasks]
        check_results(abinit_job_ids, 'Ab-initio reconstruction:', args.dry_run)
        volumes = [f'{abinit_job_id}.volume_class_0' for abinit_job_id in abinit_job_ids]

    # Homogenous / non-uniform refinement
    from itertools import product
    refine_tasks = [pool.submit(refine, session, particle, volume, args.ini_high, args.sym, args.resplit, args.nu) for (particle, volume), _ in product(zip(particles, volumes), range(args.repeat))]
    refine_job_ids = [task.result() for task in refine_tasks]
    check_results(refine_job_ids, 'Homogeneous / Non-uniform refinement:', args.dry_run)

    # Local refinement
    if args.local:
        refine_tasks = [pool.submit(local_refine, session, refine_job_id, args.sym, args.min_angular_step) for refine_job_id in refine_job_ids]
        refine_job_ids = [task.result() for task in refine_tasks]
        check_results(refine_job_ids, 'Local refinement:', args.dry_run)

    if args.dry_run:
        pool.shutdown()
        logger.info('Dry run: skip result parsing and summary CSV output')
        return

    # Get results
    parse_result_tasks = [pool.submit(parse_result, session, refine_job_id) for refine_job_id in refine_job_ids]
    results = [task.result() for task in parse_result_tasks]
    check_results(results)
    pool.shutdown()

    # Save results
    import csv
    with open(args.o, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'number of particles', 'resolution (A)', 'b-factor (A^2)', 'alignment3D'])
        for i, result in enumerate(results):
            particle_meta_path = particle_meta_paths[i // args.repeat]
            csvwriter.writerow([particle_meta_path] + result)

def main():
    args = parse_argument()
    process(args)

if __name__ == '__main__':
    main()
